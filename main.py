
import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel
from tokenizers import BPETokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def generate(seed):
	remove_eot = False
	if seed.strip() == '':
		seed = '<|endoftext|>'
		remove_eot = True
	context_tokens = tokenizer.encode(seed).ids

	out = sample_sequence(
		model=model,
		context=context_tokens,
		num_samples=args.num_samples,
		length=args.length,
		temperature=args.temperature,
		top_k=args.top_k,
		top_p=args.top_p,
		device=args.device,
	)
	
	out = out.tolist()
	samples = []
	for o in out:
		sample = tokenizer.decode(o)
		if remove_eot:
			sample = sample[len(seed)+1:]
		index = sample.find('<|endoftext|>')
		if index != -1:
			sample = sample[:index]
		
		samples.append(sample)

	result = '\n\n'.join(samples).replace('; ', '\n')
	return result

def set_seed(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, temperature=1, filter_value=-float('Inf')):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
		Args:
			logits: logits distribution shape (batch size x vocabulary size)
			top_k > 0: keep only top k tokens with highest probability (top-k filtering).
			top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
				Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
	"""
	top_k = min(top_k, logits.size(-1))  # Safety check
	if top_k > 0:
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p > 0.0:
		logits = logits/temperature
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > top_p
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0
		# scatter sorted tensors to original indexing
		indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
		logits[indices_to_remove] = filter_value
	return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0,
				 device='cpu'):
	context = torch.tensor(context, dtype=torch.long, device=device)
	context = context.unsqueeze(0).repeat(num_samples, 1)
	generated = context
	probs = np.ones((num_samples))
	lens = np.zeros(num_samples)
	has_ended = np.zeros(num_samples) != 0
	with torch.no_grad():
		
		for curr_len in trange(length):
			past = None
			inputs = {'input_ids': generated, 'past': past}
			outputs, past = model(**inputs) 
			next_token_logits = outputs[:, -1, :] / (temperature if temperature > 0 else 1.)
			next_token_probs = F.softmax(next_token_logits, dim=-1).cpu().numpy()
			filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
			
			if temperature == 0: # argmax sampling:
				next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
			else:
				next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

			next_token_np = np.squeeze(next_token.cpu().numpy())
			generated = torch.cat((generated, next_token), dim=1)
	

	return generated



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_path", type=str, required=True,
					help="Path to pre-trained model")
parser.add_argument("--tokenizer_path", type=str, required=True,
					help="Path to pre-trained tokenizer")
parser.add_argument("--length", type=int, default=100, 
					 help="Maximum length of sample")
parser.add_argument("--num_samples", type=int, default=1,
					 help="Number of samples to generate for a prompt")
parser.add_argument("--temperature", type=float, default=1.0,
					help="Softmax temperature, temperature=0 implies greedy sampling")
parser.add_argument("--top_k", type=int, default=0,
					 help="Value of k for top-k sampling, for k=0 top-k sampling is not used ")
parser.add_argument("--top_p", type=float, default=0.9,
					 help="Value of p for nucleus sampling")
parser.add_argument("--no_cuda", action='store_true',
					help="Avoid using CUDA when available")
parser.add_argument('--seed', type=int, default=2020,
					help="Random seed for initialization")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = torch.cuda.device_count()

set_seed(args)

tokenizer = BPETokenizer(vocab_file=f'{args.tokenizer_path}/vocab.json', merges_file=f'{args.tokenizer_path}/merges.txt')
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model.to(args.device)
model.eval()
eot_token = tokenizer.encode('<|endoftext|>').ids
assert len(eot_token) == 1
eot_token = eot_token[0]


if args.length < 0 and model.config.max_position_embeddings > 0:
	args.length = model.config.max_position_embeddings
elif 0 < model.config.max_position_embeddings < args.length:
	args.length = model.config.max_position_embeddings  # No generation bigger than model size 
elif args.length < 0:
	args.length = MAX_LENGTH  # avoid infinite loop

logger.info(args)

while True:
	try:
		prompt = input('>>> ')
		print(generate(prompt), end='\n\n')
	except KeyboardInterrupt:
		print('\nQuitting')
		break

