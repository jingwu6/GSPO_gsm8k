from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class GSPOConfig:
    """Configuration for GSPO training parameters"""
    def __init__(self):
        # Model configuration
        self.model_path = "Qwen/Qwen2.5-7B"
        self.gen_device = 6  # Dedicated GPU 6 for generation (GPU 7 for ref_server)
        
        # Training hyperparameters (optimized for 8x A6000 setup)
        self.beta = 0.04                    # KL penalty coefficient
        self.all_steps = 200             # Total training steps
        self.Q_batch_size = 8              # Questions per batch (increased for better GPU utilization)
        self.num_pre_Q = 8                 # Responses per question
        self.train_batch_size = 4          # Training batch size (increased for A6000)
        self.gen_update_steps = 16         # Generator update frequency
        self.save_steps = 50               # Model save frequency
        self.test_freq = 10                # Test evaluation frequency
        self.compute_gen_logps = True      # Compute generation log probabilities
        self.clip_param = 0.2              # PPO clipping parameter
        
        # Server configuration
        self.ref_server = "http://localhost:59875"
        
        # DeepSpeed configuration (optimized for 6x A6000 training GPUs)
        self.ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": 8,  # Reduced since we have more GPUs
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-6}
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,  # Increased for A6000
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,     # Increased for A6000
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save": True,
                "offload_optimizer": {"device": "cpu"}
            }
        }

class DataManager:
    """Handles batch data communication with reference server"""
    def __init__(self, config):
        self.config = config
        try:
            from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
            self.tensor_to_bytes = tensor_to_bytes
            self.bytes_to_tensor = bytes_to_tensor
            self.make_bytes_list = make_bytes_list
            self.bytes_list_to_list = bytes_list_to_list
        except ImportError:
            print("Warning: ref_server module not found, using dummy functions")
            self.tensor_to_bytes = lambda x: x.numpy().tobytes()
            self.bytes_to_tensor = lambda x: torch.frombuffer(x, dtype=torch.float32)
            self.make_bytes_list = lambda x: b''.join(x)
            self.bytes_list_to_list = lambda x: [x]
    
    def get_batch(self):
        """Fetch training batch from reference server"""
        try:
            r = requests.get(f"{self.config.ref_server}/get", timeout=5.0)
            if r.status_code != 200:
                print(f"[DATA] Server returned status {r.status_code}")
                return None
            if r.content == b'empty': 
                return None
            data_list = self.bytes_list_to_list(r.content)
            batch_info = json.loads(data_list[0].decode())
            merged_ids = self.bytes_to_tensor(data_list[1])
            rewards = self.bytes_to_tensor(data_list[2])
            
            batch = {
                'input_ids': merged_ids,
                'rewards': rewards,
                'plen': batch_info['plen'],
                'is_test': batch_info.get('is_test', False)
            }
            
            # Add generation log probabilities if available
            if len(data_list) > 3:
                batch['gen_logps'] = self.bytes_to_tensor(data_list[3])
            
            # Add separate reward components if available
            if len(data_list) > 4:
                batch['correct_rewards'] = self.bytes_to_tensor(data_list[4])
            if len(data_list) > 5:
                batch['format_rewards'] = self.bytes_to_tensor(data_list[5])
            
            return batch
        except Exception as e:
            if "Connection refused" in str(e):
                print(f"[DATA] Cannot connect to ref_server at {self.config.ref_server}")
            else:
                print(f"[DATA] Error getting batch: {e}")
            return None
    
    def get_test_batch(self):
        """Fetch test batch from reference server"""
        try:
            r = requests.get(f"{self.config.ref_server}/get_test", timeout=5.0)
            if r.content == b'None': 
                return None
            data_list = self.bytes_list_to_list(r.content)
            batch_info = json.loads(data_list[0].decode())
            merged_ids = self.bytes_to_tensor(data_list[1])
            rewards = self.bytes_to_tensor(data_list[2])
            
            batch = {
                'input_ids': merged_ids,
                'rewards': rewards,
                'plen': batch_info['plen'],
                'is_test': True
            }
            
            if len(data_list) > 3:
                batch['gen_logps'] = self.bytes_to_tensor(data_list[3])
            
            return batch
        except Exception as e:
            print(f"Warning: Failed to get test batch from server: {e}")
            return None

class GSPOModel:
    """GSPO model implementation with policy optimization"""
    def __init__(self, config):
        self.config = config
        
    def get_per_token_logps(self, logits, input_ids):
        """Calculate per-token log probabilities from logits"""
        logps = torch.log_softmax(logits, dim=-1)
        logps = torch.gather(logps, 2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
        return logps
    
    def gspo_step(self, batch, engine):
        # Move tensors to the correct device
        device = next(engine.module.parameters()).device
        input_ids = batch['input_ids'].to(device)
        rewards = batch['rewards'].to(device) 
        gen_logps = batch.get('gen_logps', None)
        if gen_logps is not None:
            gen_logps = gen_logps.to(device)
        
        plen = batch['plen']
        
        # Forward pass through model
        outputs = engine.module(input_ids)
        logits = outputs.logits
        
        # Calculate current policy log probabilities
        logps = self.get_per_token_logps(logits, input_ids)
        
        # Use reference log probabilities if available, otherwise use current
        if gen_logps is not None:
            ref_logps = gen_logps
        else:
            ref_logps = logps.detach()
        
        # Calculate sequence-level log probabilities (sum over tokens)
        seq_logps = logps[:, plen-1:].sum(dim=1)
        ref_seq_logps = ref_logps[:, plen-1:].sum(dim=1)
        
        # Calculate importance ratios for policy gradient
        log_ratios = seq_logps - ref_seq_logps
        ratios = torch.exp(log_ratios)
        
        # Apply PPO-style clipping to ratios
        clipped_ratios = torch.clamp(ratios, 1 - self.config.clip_param, 1 + self.config.clip_param)
        
        # GSPO loss: minimize negative expected reward with clipping
        policy_loss = -torch.min(ratios * rewards, clipped_ratios * rewards).mean()
        
        # Add KL divergence penalty to prevent policy drift
        kl_penalty = self.config.beta * log_ratios.mean()
        
        total_loss = policy_loss + kl_penalty
        
        return total_loss

class MetricsLogger:
    """Handles logging of training and evaluation metrics"""
    def __init__(self, config):
        self.config = config
        self.log_file = './logs/training_log.jsonl'
        os.makedirs('./logs', exist_ok=True)
    
    def log_metrics(self, step, loss=None, batch=None, metrics_type='train'):
        """Log metrics to JSONL file and return computed metrics"""
        if batch is not None:
            rewards = batch['rewards'].cpu().numpy()
            correct_rewards = batch.get('correct_rewards', torch.zeros_like(batch['rewards'])).cpu().numpy()
            format_rewards = batch.get('format_rewards', torch.zeros_like(batch['rewards'])).cpu().numpy()
            
            avg_reward = float(rewards.mean())
            acc_ratio = float((rewards > 0).mean())      # Combined accuracy
            format_ratio = float((rewards > 1).mean())   # Format accuracy
            math_accuracy = float((correct_rewards > 0).mean())    # Pure math correctness
            format_accuracy = float((format_rewards > 0).mean())   # Pure format correctness
        else:
            avg_reward = acc_ratio = format_ratio = math_accuracy = format_accuracy = 0.0
        
        metrics = {
            'step': step,
            'type': metrics_type,
            'acc_ratio': acc_ratio,
            'format_ratio': format_ratio,
            'math_accuracy': math_accuracy,
            'format_accuracy': format_accuracy,
            'avg_reward': avg_reward
        }
        
        if loss is not None:
            metrics['loss'] = float(loss.item())
        
        if batch is not None and 'rewards' in batch:
            metrics['test_samples'] = len(batch['rewards'])
        
        # Write to JSONL log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        return metrics

class GSPOTrainer:
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model = GSPOModel(config)
        self.logger = MetricsLogger(config)
        self.Q = None
        self.p = None
    
    def setup_model(self):
        """Initialize model, tokenizer, and DeepSpeed engine"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa"
        )
        
        import deepspeed
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            config=self.config.ds_config,
            model=model,
            model_parameters=model.parameters()
        )
    
    def setup_generation_worker(self):
        """Start generation worker process for vLLM inference"""
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('[GSPO] Starting vLLM generation worker...')
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                print('[GSPO] Multiprocessing start method already set')
            
            self.Q = mp.Queue(maxsize=5)
            self.p = mp.Process(target=gen_worker, args=(
                self.Q, 
                self.config.gen_device,
                self.config.model_path,
                self.config.Q_batch_size,
                self.config.num_pre_Q,
                self.config.train_batch_size,
                self.config.compute_gen_logps
            ))
            self.p.start()
            print('[GSPO] Generation worker started successfully')
    
    def train(self):
        """Main training loop with periodic evaluation and model saving"""
        progress = range(1, self.config.all_steps + 1)
        if dist.get_rank() == 0:
            progress = tqdm(progress)
        
        batch_wait_count = 0
        wait_time = 1.0
        max_wait_attempts = 100  # Prevent infinite waiting
        
        for step in progress:
            batch = None
            
            # Improved batch retrieval with exponential backoff and max attempts
            while batch is None and batch_wait_count < max_wait_attempts:
                batch_wait_count += 1
                if batch_wait_count % 20 == 1:
                    print(f'[TRAIN] Waiting for batch... (attempt {batch_wait_count})')
                time.sleep(min(wait_time, 5.0))  # Cap wait time
                batch = self.data_manager.get_batch()
                
                if batch is None:
                    wait_time = min(wait_time * 1.1, 5.0)  # Exponential backoff with cap
            
            if batch is None:
                print(f'[TRAIN] Failed to get batch after {max_wait_attempts} attempts, skipping step {step}')
                continue
            
            if batch_wait_count > 0:
                batch_wait_count = 0
                wait_time = 1.0  # Reset wait time
            
            # Perform GSPO training step
            loss = self.model.gspo_step(batch, self.engine)
            self.engine.backward(loss)
            self.engine.step()
            
            if dist.get_rank() == 0:
                progress.set_description(f"Loss: {loss.item():.6f}")
                
                # Log training metrics every 5 steps
                if step % 5 == 0:
                    enhanced_batch = {
                        'rewards': batch['rewards'],
                        'correct_rewards': batch.get('correct_rewards', torch.zeros_like(batch['rewards'])),
                        'format_rewards': batch.get('format_rewards', torch.zeros_like(batch['rewards']))
                    }
                    
                    metrics = self.logger.log_metrics(step, loss, enhanced_batch, 'train')
                    print(f"📊 Step {step}: Loss={loss.item():.6f}, Math={metrics['math_accuracy']:.3f}, Format={metrics['format_accuracy']:.3f}, Combined={metrics['acc_ratio']:.3f}, Reward={metrics['avg_reward']:.3f}")
                
                # Run test evaluation every test_freq steps
                if step % self.config.test_freq == 0:
                    self.run_test_evaluation(step)
            
            # Update generation model weights
            if step % self.config.gen_update_steps == 0:
                dist.barrier()
                if dist.get_rank() == 0:
                    state_dict = self.engine.module.state_dict()
                    try:
                        if self.Q is not None:
                            self.Q.put(state_dict, timeout=5.0)
                    except Exception as e:
                        print(f'[TRAIN] Warning: Failed to send model update at step {step}: {e}')
                dist.barrier()
            
            # Save model checkpoint
            if step % self.config.save_steps == 0:
                dist.barrier()
                if dist.get_rank() == 0:
                    save_name = f"./step_{step}"
                    state_dict = self.engine.module.state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    self.engine.module.save_pretrained(save_name, state_dict=state_dict)
                    self.tokenizer.save_pretrained(save_name)
                    print(f'[TRAIN] Model saved at step {step}')
                dist.barrier()
    
    def run_test_evaluation(self, step):
        """Run test evaluation with improved metrics"""
        if step % self.config.test_freq == 0:
            print(f"🧪 Running test evaluation at step {step}...")
            test_rewards = []
            test_correct_rewards = []
            test_format_rewards = []
            test_batches_collected = 0
            target_test_batches = 100
            
            while test_batches_collected < target_test_batches:
                test_batch = self.data_manager.get_test_batch()
                if test_batch is None:
                    break
                
                test_rewards.extend(test_batch['rewards'].cpu().numpy())
                if 'correct_rewards' in test_batch:
                    test_correct_rewards.extend(test_batch['correct_rewards'].cpu().numpy())
                    test_format_rewards.extend(test_batch['format_rewards'].cpu().numpy())
                
                test_batches_collected += 1
            
            if test_rewards:
                import numpy as np
                test_rewards = np.array(test_rewards)
                
                enhanced_test_batch = {
                    'rewards': torch.tensor(test_rewards),
                    'correct_rewards': torch.tensor(test_correct_rewards) if test_correct_rewards else torch.zeros_like(torch.tensor(test_rewards)),
                    'format_rewards': torch.tensor(test_format_rewards) if test_format_rewards else torch.zeros_like(torch.tensor(test_rewards))
                }
                
                metrics = self.logger.log_metrics(step, batch=enhanced_test_batch, metrics_type='test')
                print(f"🎯 Test Step {step}: Math={metrics['math_accuracy']:.3f}, Format={metrics['format_accuracy']:.3f}, Combined={metrics['acc_ratio']:.3f}, Reward={metrics['avg_reward']:.3f} (n={len(test_rewards)})")
                
                return metrics
        return None
    
    def cleanup(self):
        """Clean up generation worker process"""
        if self.p is not None and self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=5.0)
            if self.p.is_alive():
                self.p.kill()

def gen_worker(Q, physics_device=1, model_path=None, Q_batch_size=None, num_pre_Q=None, train_batch_size=None, compute_gen_logps=None):
    """Generation worker process using vLLM for inference"""
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from datasets import load_dataset
        import math
        from torch.nn.utils.rnn import pad_sequence
        
        # Try to import ref_server functions
        try:
            from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
        except ImportError:
            print("Warning: ref_server module not found, using dummy functions")
            def tensor_to_bytes(x): return x.numpy().tobytes()
            def bytes_to_tensor(x): return torch.frombuffer(x, dtype=torch.float32)
            def make_bytes_list(x): return b''.join(x)
            def bytes_list_to_list(x): return [x]
        
        # Try to import math_verify
        try:
            from math_verify import parse, verify, ExprExtractionConfig
        except ImportError:
            print("Warning: math_verify not available, using simple numerical comparison")
            def parse(x, **kwargs): return x
            def verify(x, y): return str(x).strip() == str(y).strip()
            ExprExtractionConfig = lambda: None
        
        # Initialize tokenizer within the worker process
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        ref_server = "http://localhost:59875"
        ref_server_ver = 'tensor'
        
        # Set GPU device for generation - use dedicated GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(physics_device)
        print(f"[GEN WORKER] Using dedicated GPU {physics_device} for generation")
        
        # Initialize vLLM engine optimized for A6000 (48GB VRAM)
        try:
            print(f"[GEN WORKER] Initializing vLLM on dedicated GPU {physics_device}")
            # Around line 449 in the gen_worker function
            vllm_gen = LLM(
            model=model_path, 
            gpu_memory_utilization=0.5,  # Changed from 0.85 to 0.5
            max_model_len=8192,           
            dtype="bfloat16"
            )
            gen_logps_sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
            print(f"[GEN WORKER] vLLM initialization successful")
        except Exception as e:
            print(f"[GEN WORKER] Failed to initialize vLLM: {e}")
            return
        
        # Load GSM8K datasets
        try:
            print(f"[GEN WORKER] Loading GSM8K datasets...")
            gsm8k = load_dataset("openai/gsm8k", "main", split="train")
            QAs = [(x['question'], x['answer']) for x in gsm8k]
            
            test_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
            test_QAs = [(x['question'], x['answer']) for x in test_gsm8k]
            print(f"[GEN WORKER] Loaded {len(QAs)} training and {len(test_QAs)} test samples")
        except Exception as e:
            print(f"[GEN WORKER] Failed to load datasets: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Define system prompt
        system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        
        def gen_answers(questions):
            """Generate answers for given questions using vLLM"""
            tip_text = []
            for x in questions:
                tip_text.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
            
            sp = SamplingParams(temperature=0.7, max_tokens=512, stop=['Question:', 'Q:'], n=num_pre_Q)
            voutputs = vllm_gen.generate(tip_text, sp, use_tqdm=False)
            
            answers = []
            ans_token_ids = []
            for v in voutputs:
                for z in v.outputs: 
                    answers.append(z.text)
                    ans_token_ids.append(z.token_ids)
            
            return answers, ans_token_ids
        
        def reward_correct(item, answer):
            """Calculate correctness reward using math_verify library"""
            pattern = r'\d+\.\d+|\d+/\d+|\d+'
            nums = re.findall(pattern, answer) 
            if len(nums) == 0: return -1.0
            lastnum = nums[-1]
            try:
                ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
                ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
                return 1 if verify(ans, ground_truth) else -1
            except:
                return -1
        
        def reward_format(item, answer):
            """Calculate format reward for <think></think><answer></answer> structure"""
            pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
            think_count = answer.count("<think>") + answer.count("</think>")
            answer_count = answer.count("<answer>") + answer.count("</answer>")
            return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1
        
        def gen_samples(inputs):
            """Generate samples and calculate rewards for training"""
            prompts = [x["Q"] for x in inputs]
            answers, ans_token_ids = gen_answers(prompts)
            rewards = []
            correct_rewards = []
            format_rewards = []
            
            for i, inp in enumerate(inputs):
                for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                    correct_r = reward_correct(inp, a)
                    format_r = reward_format(inp, a)
                    rewards.append(correct_r + format_r)
                    correct_rewards.append(correct_r)
                    format_rewards.append(format_r)
            
            prompts_text = [tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
            
            return (prompts_text, torch.tensor(rewards, dtype=torch.float32), 
                    torch.tensor(correct_rewards, dtype=torch.float32),
                    torch.tensor(format_rewards, dtype=torch.float32), answers, ans_token_ids)
        
        def try_update_model():
            """Update vLLM model with new weights from training process"""
            try:
                new_state_dict = Q.get_nowait()
                print('[GEN PROC] received state_dict, updating vLLM model...')
                vllm_gen.llm_engine.model_executor.load_model()
                del new_state_dict
            except:
                return
        
        # Main generation loop
        print(f"[GEN WORKER] Starting generation loop, connecting to {ref_server}")
        for it in range(999999999):
            if it % 10 == 0:
                print(f"[GEN WORKER] Generation iteration {it}")
            
            # Periodically update model weights
            if it % 3 == 0: 
                try_update_model()
            
            # Alternate between test and train data
            is_test_batch = it % 5 == 0
            inputs = [{"Q": q, "A": a} for q, a in (random.sample(test_QAs, Q_batch_size) if is_test_batch else random.sample(QAs, Q_batch_size))]
            tic = time.time()
            
            print(f"[GEN WORKER] Generating samples for iteration {it}, test_batch={is_test_batch}")
            
            # Fixed unpacking of gen_samples result
            try:
                result = gen_samples(inputs)
                if len(result) == 6:
                    prompt_inputs, rewards, correct_rewards, format_rewards, answers, ans_token_ids = result
                elif len(result) == 4:
                    prompt_inputs, rewards, answers, ans_token_ids = result
                    # Create dummy separate rewards if not available
                    correct_rewards = torch.zeros_like(rewards)
                    format_rewards = torch.zeros_like(rewards)
                else:
                    raise ValueError(f"Unexpected return from gen_samples: {len(result)} values")
            except ValueError as e:
                print(f"Error unpacking gen_samples result: {e}")
                continue
            
            batch_type = "[TEST]" if is_test_batch else "[TRAIN]"
            print(f'{batch_type} time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
            if it % 5 == 0: 
                print('answers:', answers[0])
            
            # Process each question's responses
            for i, pp in enumerate(prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_correct_rewards = correct_rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_format_rewards = format_rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                
                # Skip if rewards have no variance
                if curr_rewards.max() - curr_rewards.min() < 1e-4: 
                    continue
                
                # Send data to reference server
                if ref_server_ver == 'tensor':
                    # Normalize rewards for group advantage
                    curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                    for ii in range(0, num_pre_Q, train_batch_size):
                       sub_rewards = curr_rewards[ii:ii+train_batch_size]
                        sub_correct_rewards = curr_correct_rewards[ii:ii+train_batch_size]
                        sub_format_rewards = curr_format_rewards[ii:ii+train_batch_size]
                        
                        data = [json.dumps(base_data).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]
                        
                        # Add separate reward components
                        if len(curr_correct_rewards) > 0:
                            data.append(tensor_to_bytes(sub_correct_rewards))
                            data.append(tensor_to_bytes(sub_format_rewards))
                        
                        if compute_gen_logps:
                            data.append(tensor_to_bytes(gen_logps))
                            try:
                                zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), 
                                                     sampling_params=gen_logps_sp, use_tqdm=False)
                                zz = [xx.prompt_logprobs[plen:] for xx in zz]
                                gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                                data.append(tensor_to_bytes(gen_logps))
                            except Exception as e:
                                print(f"Warning: Failed to compute generation logps: {e}")
                        
                        try:
                            xdata = make_bytes_list(data)
                            r = requests.post(f"{ref_server}/upload", data=xdata, timeout=5.0)
                            print(f"[GEN WORKER] Sent batch to server, response: {r.status_code}")
                            if r.content == b'string': 
                                ref_server_ver = 'string'
                        except Exception as e:
                            print(f"Warning: Failed to upload batch to server: {e}")
                            
                elif ref_server_ver == 'string':
                    try:
                        xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                               tensor_to_bytes(curr_rewards)])
                        r = requests.post(f"{ref_server}/upload", data=xdata, timeout=5.0)
                        if r.content == b'tensor': 
                            ref_server_ver = 'tensor'
                    except Exception as e:
                        print(f"Warning: Failed to upload string batch to server: {e}")
    
    except Exception as e:
        print(f"Fatal error in generation worker: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import deepspeed
    import socket, contextlib
    import torch.distributed as dist
    
    def _find_free_port():
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    config = GSPOConfig()
    trainer = None
    
    try:
        # Set a default MASTER_PORT
        os.environ['MASTER_PORT'] = '29500'
        
        # Initialize distributed
        deepspeed.init_distributed()
        
        # Only rank 0 finds a free port if needed
        if dist.get_rank() == 0:
            current_port = int(os.environ.get('MASTER_PORT', '29500'))
            if _port_in_use(current_port):
                new_port = _find_free_port()
                os.environ['MASTER_PORT'] = str(new_port)
                print(f"[GSPO] MASTER_PORT {current_port} busy, switched to free port {new_port}.")
        
        # Broadcast the final MASTER_PORT to all ranks
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{dist.get_rank() % torch.cuda.device_count()}')
        else:
            device = torch.device('cpu')
            
        master_port = torch.tensor([int(os.environ['MASTER_PORT'])], dtype=torch.int, device=device)
        dist.broadcast(master_port, src=0)
        os.environ['MASTER_PORT'] = str(master_port.item())
        
        # Setup and run training
        trainer = GSPOTrainer(config)
        trainer.setup_model()
        trainer.setup_generation_worker()
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n[GSPO] Training interrupted by user")
    except Exception as e:
        print(f"[GSPO] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if trainer is not None:
            trainer.cleanup()
        print("[GSPO] Cleanup completed")