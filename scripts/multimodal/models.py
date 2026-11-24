"""Model-specific preprocessing shared by inference and training."""

import math
from pathlib import Path


def load_model(cfg, adapter=None, training=False):
    import torch
    from transformers import AutoProcessor, ClapModel, MusicFlamingoForConditionalGeneration, Qwen3ASRForConditionalGeneration, Qwen3_5ForConditionalGeneration

    classes = {'vision': Qwen3_5ForConditionalGeneration,
               'speech': Qwen3ASRForConditionalGeneration, 'audio': ClapModel,
               'music': MusicFlamingoForConditionalGeneration}
    source = cfg['model_dir']
    if adapter and cfg['task'] == 'audio':
        source = str(adapter)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    dtype = getattr(torch, cfg['dtype'])
    kwargs = {'dtype': dtype, 'local_files_only': True, 'device_map': cfg['device']}
    if cfg['task'] != 'audio':
        kwargs['attn_implementation'] = 'sdpa'
    model = classes[cfg['task']].from_pretrained(source, **kwargs)
    if adapter and cfg['task'] != 'audio':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=training)
    elif training and cfg['task'] != 'audio':
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg['lora_rank'], lora_alpha=2 * cfg['lora_rank'],
            target_modules=['q_proj', 'v_proj'], lora_dropout=0.05,
        ))
    if training and cfg['task'] == 'audio':
        # Adapt the shared embedding space while preserving the two encoders.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(('audio_projection.', 'text_projection.', 'logit_scale'))
    model.train(training)
    return model, processor


def audio_array(path, sample_rate):
    import librosa
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must contain finite samples')
    if rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    return audio


def conversation(row, task):
    if task == 'vision':
        return [{'role': 'user', 'content': [
            {'type': 'image', 'url': row['image']},
            {'type': 'text', 'text': row.get('prompt', 'Describe the image.')}]}]
    if task == 'music':
        return [{'role': 'user', 'content': [
            {'type': 'text', 'text': row.get('prompt', 'Describe the musical instruments, genre, rhythm and mood you hear.')},
            {'type': 'audio', 'path': row['audio']}]}]
    return [{'role': 'user', 'content': [{'type': 'audio', 'path': row['audio']}]}]


class Collator:
    def __init__(self, cfg, processor):
        self.cfg = cfg
        self.processor = processor

    def __call__(self, rows):
        task = self.cfg['task']
        processor = self.processor
        if task == 'audio':
            rate = processor.feature_extractor.sampling_rate
            audio = [audio_array(row['audio'], rate) for row in rows]
            if any(len(item) > 10 * rate for item in audio):
                raise ValueError('CLAP training pairs must be clips of at most 10 seconds')
            return processor(audio=audio, text=[row['text'] for row in rows],
                             sampling_rate=rate, padding=True, return_tensors='pt')
        row = rows[0]
        messages = conversation(row, task)
        prompt = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt', enable_thinking=False,
        )
        answer = row['text']
        if task == 'speech':
            answer = f"language {row['language']}<asr_text>{answer}"
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors='pt', enable_thinking=False,
        )
        prompt_length = prompt['input_ids'].shape[1]
        import torch
        if not torch.equal(prompt['input_ids'][0], inputs['input_ids'][0, :prompt_length]):
            raise ValueError('Chat template prefix differs between prompt and target')
        labels = inputs['input_ids'].clone()
        labels[:, :prompt_length] = -100
        labels[inputs['attention_mask'] == 0] = -100
        if not (labels != -100).any():
            raise ValueError('Training example has no target tokens')
        if inputs['input_ids'].shape[1] > self.cfg.get('max_tokens', 4096):
            raise ValueError('Example exceeds max_tokens; shorten the sample or raise the limit')
        inputs['labels'] = labels
        return inputs


def infer(cfg, path, prompt=None, labels=None, adapter=None):
    import torch
    import torch.nn.functional as F

    model, processor = load_model(cfg, adapter)
    task = cfg['task']
    with torch.inference_mode():
        if task == 'audio':
            rate = processor.feature_extractor.sampling_rate
            waveform = audio_array(path, rate)
            vectors = []
            for start in range(0, len(waveform), 10 * rate):
                inputs = processor(audio=waveform[start:start + 10 * rate],
                                   sampling_rate=rate, return_tensors='pt').to(model.device)
                features = model.get_audio_features(**inputs)
                if hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                vectors.append(F.normalize(features.float(), dim=-1))
            vector = F.normalize(torch.cat(vectors).mean(dim=0, keepdim=True), dim=-1)
            result = {'embedding': vector[0].cpu().tolist(), 'dimensions': vector.shape[-1],
                      'segments': len(vectors), 'seconds': len(waveform) / rate}
            if labels:
                inputs = processor(text=labels, padding=True, return_tensors='pt').to(model.device)
                text = model.get_text_features(**inputs)
                if hasattr(text, 'pooler_output'):
                    text = text.pooler_output
                scores = (vector @ F.normalize(text.float(), dim=-1).T)[0].cpu().tolist()
                result['similarities'] = dict(zip(labels, scores))
            if not all(math.isfinite(x) for x in result['embedding']):
                raise RuntimeError('Nonfinite audio embedding')
            return result
        if task in ('vision', 'music'):
            key = 'image' if task == 'vision' else 'audio'
            row = {key: str(Path(path).resolve())}
            if prompt:
                row['prompt'] = prompt
            messages = conversation(row, task)
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors='pt', enable_thinking=False)
        else:
            inputs = processor.apply_transcription_request(audio=str(path), language=cfg.get('language'))
        inputs = inputs.to(model.device, model.dtype)
        output = model.generate(**inputs, max_new_tokens=cfg.get('max_new_tokens', 256), do_sample=False)
        generated = output[:, inputs['input_ids'].shape[1]:]
        if task == 'speech':
            text = processor.decode(generated[0], return_format='transcription_only')
        else:
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        if not text.strip():
            raise RuntimeError('Model returned empty text')
        return {'text': text}
