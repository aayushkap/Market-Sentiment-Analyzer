from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from pathlib import Path
from typing import Generator, Optional
from config import model_dir, device, low_cpu_mem_usage, llm_model_name


class LLMSummarizer:
	def __init__(
		self,
		model_name: str = llm_model_name,
		device: str = device,
		dtype=torch.float32,
		low_cpu_mem_usage: bool = low_cpu_mem_usage,
		model_dir: str = f"{model_dir}/llm",
	):
		self.model_name = model_name
		self.device = device
		self.dtype = dtype

		self.tokenizer = None
		self.model = None

		self.model_dir = Path(model_dir)

		self.load_model(low_cpu_mem_usage)

	def load_model(self, low_cpu_mem_usage: bool):
		if self.model_dir.exists() and (self.model_dir / "config.json").exists():
			print(f"Loading model from local directory: {self.model_dir}")
			model_path = str(self.model_dir)
		else:
			print(f"Model not found locally. Downloading to: {self.model_dir}")
			self.model_dir.mkdir(parents=True, exist_ok=True)
			model_path = self.model_name

		self.tokenizer = AutoTokenizer.from_pretrained(
			model_path,
			cache_dir=(str(self.model_dir) if model_path == self.model_name else None),
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_path,
			dtype=self.dtype,
			device_map=self.device,
			low_cpu_mem_usage=low_cpu_mem_usage,
			cache_dir=(str(self.model_dir) if model_path == self.model_name else None),
		)

		if model_path == self.model_name:
			print(f"Saving model to: {self.model_dir}")
			self.tokenizer.save_pretrained(str(self.model_dir))
			self.model.save_pretrained(str(self.model_dir))

	def summarize_stream(
		self,
		prompt: str,
		max_tokens: int = 150,
		temperature: float = 0.3,
		top_p: float = 0.6,
		do_sample: bool = True,
		repetition_penalty: float = 1.0,
	) -> Generator[str, None, None]:
		messages = [
			{"role": "system", "content": "You are a financial analyst providing brief, factual market summaries."},
			{"role": "user", "content": prompt}
		]
		
		formatted_prompt = self.tokenizer.apply_chat_template(
			messages, 
			tokenize=False, 
			add_generation_prompt=True
		)
		
		inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

		streamer = TextIteratorStreamer(
			self.tokenizer, skip_prompt=True, skip_special_tokens=True
		)

		generation_kwargs = {
			**inputs,
			"max_new_tokens": max_tokens,
			"temperature": temperature,
			"top_p": top_p,
			"do_sample": do_sample,
			"repetition_penalty": repetition_penalty,
			"pad_token_id": self.tokenizer.eos_token_id,
			"eos_token_id": self.tokenizer.eos_token_id,
			"streamer": streamer,
		}

		thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
		thread.start()

		for token in streamer:
			yield token

		thread.join()

	def __repr__(self):
		return f"LLMSummarizer(model='{self.model_name}', device='{self.device}')"
