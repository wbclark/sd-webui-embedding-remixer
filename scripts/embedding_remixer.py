import gradio as gr
from modules import script_callbacks, shared, devices, sd_hijack, sd_samplers, scripts, processing, images
from modules.shared import cmd_opts, opts
from modules.processing import process_images, Processed
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import collections, math, random, re
from modules.processing import process_images, StableDiffusionProcessingTxt2Img
from PIL import Image
import json
from contextlib import closing
from safetensors.torch import load_file, save_file

MAX_NUM_MIX = 75
SHOW_NUM_MIX = 6
MAX_SIMILAR_EMBS = 100
VEC_SHOW_THRESHOLD = 1 # change to 10000 to see all values
VEC_SHOW_PROFILE = 'default' #change to 'full' for more precision
SEP_STR = '-'*80

SHOW_SIMILARITY_SCORE = False

ENABLE_SHOW_CHECKSUM = False

EVAL_PRESETS = ['None','',
    'Boost','=v*8',
    'Digitize','=math.ceil(v*8)/8',
    'Binary','=(1*(v>=0)-1*(v<0))/50',
    'Randomize','=v*random.random()',
    'Sine','=v*math.sin(i/maxi*math.pi)',
    'Comb','=v*((i%2)==0)',
    'Crop_high','=v*(i<maxi//2)',
    'Crop_low','=v*(i>=maxi//2)',
    'Max_pool', '=tot_vec[torch.argmax(torch.abs(tot_vec[:, i])), i]',
    'Max_pool_L2', '=tot_vec[torch.argmax(torch.abs(tot_vec[:, i]) / torch.norm(tot_vec, dim=1, keepdim=True)[:, 0]), i]',
    'Soft Attention', '=(torch.softmax(tot_vec @ tot_vec.T, dim=1) @ tot_vec)[n]'
    ]

def load_embedding(file_path):
    if file_path.endswith('.safetensors'):
        data = safetensors.torch.load_file(file_path, device="cpu")
    elif file_path.endswith('.pt'):
        data = torch.load(file_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if "clip_l" in data and "clip_g" in data:
        return (data["clip_l"], data["clip_g"])
    elif "string_to_param" in data:
        return data["string_to_param"]
    else:
        raise ValueError(f"Unknown embedding format in {file_path}")

def get_data():
    loaded_embs = collections.OrderedDict(
        sorted(
            sd_hijack.model_hijack.embedding_db.word_embeddings.items(),
            key=lambda x: str(x[0]).lower()
        )
    )

    if hasattr(shared.sd_model, 'conditioner'):
        embedders = shared.sd_model.conditioner.embedders
        clip_l = next((e for e in embedders if type(e).__name__ == 'CLIP_SD_XL_L'), None)
        clip_g = next((e for e in embedders if type(e).__name__ == 'CLIP_SD_XL_G'), None)

        tokenizer = clip_l.tokenizer
        internal_embs_l = clip_l.wrapped.transformer.text_model.embeddings.token_embedding.weight
        internal_embs_g = clip_g.wrapped.transformer.text_model.embeddings.token_embedding.weight

        return tokenizer, (internal_embs_l, internal_embs_g), loaded_embs

    elif hasattr(shared.sd_model, 'cond_stage_model'):
        embedder = shared.sd_model.cond_stage_model.wrapped
        if embedder.__class__.__name__=='FrozenCLIPEmbedder':
            tokenizer = embedder.tokenizer
            internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

        elif embedder.__class__.__name__=='FrozenOpenCLIPEmbedder':
            from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
            tokenizer = open_clip_tokenizer
            internal_embs = embedder.model.token_embedding.wrapped.weight

    else:
        tokenizer = None
        internal_embs = None

    return tokenizer, internal_embs, loaded_embs

def text_to_emb_ids(text, tokenizer):
    text = text.lower()

    if tokenizer.__class__.__name__ == 'CLIPTokenizer':
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    elif tokenizer.__class__.__name__ == 'SimpleTokenizer':
        emb_ids = tokenizer.encode(text)

    else:
        emb_ids = None

    return emb_ids

def emb_id_to_name(emb_id, tokenizer):

    emb_name_utf8 = tokenizer.decoder.get(emb_id)

    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else:
        emb_name = '!Unknown ID!'

    return emb_name

def get_embedding_info(text):
    text = text.lower()

    tokenizer, internal_embs, loaded_embs = get_data()

    loaded_emb = loaded_embs.get(text, None)

    if loaded_emb == None:
        for k in loaded_embs.keys():
            if text == k.lower():
                loaded_emb = loaded_embs.get(k, None)
                break

    if loaded_emb!=None:
        emb_name = loaded_emb.name
        emb_id = '['+loaded_emb.checksum()+']'
        emb_vec = loaded_emb.vec
        return emb_name, emb_id, emb_vec, loaded_emb

    val = None
    if text.startswith('#'):
        try:
            val = int(text[1:])
            if isinstance(internal_embs, tuple):
                if (val<0) or (val>=internal_embs[0].shape[0]): val = None
            else:
                if (val<0) or (val>=internal_embs.shape[0]): val = None
        except:
            val = None

    if val!=None:
        emb_id = val
    else:
        emb_ids = text_to_emb_ids(text, tokenizer)
        if len(emb_ids)==0: return None, None, None, None
        emb_id = emb_ids[0]

    emb_name = emb_id_to_name(emb_id, tokenizer)

    if isinstance(internal_embs, tuple):
        emb_vec_l = internal_embs[0][emb_id].unsqueeze(0)
        emb_vec_g = internal_embs[1][emb_id].unsqueeze(0)
        emb_vec = (emb_vec_l, emb_vec_g)
    else:
        emb_vec = internal_embs[emb_id].unsqueeze(0)

    return emb_name, emb_id, emb_vec, None

def do_inspect(text):
    text = text.strip().lower()
    if (text==''): return 'Need embedding name or embedding ID as #nnnnn', None

    emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(text)
    if (emb_name==None) or (emb_id==None) or (emb_vec==None):
        return 'An error occurred', None

    results = []

    results.append(f'Embedding name: "{emb_name}"')
    results.append(f'Embedding ID: {emb_id} ({"internal" if isinstance(emb_id, int) else "loaded"})')

    if loaded_emb!=None:
        results.append(f'Step: {loaded_emb.step}')
        results.append(f'SD checkpoint: {loaded_emb.sd_checkpoint}')
        results.append(f'SD checkpoint name: {loaded_emb.sd_checkpoint_name}')
        if hasattr(loaded_emb, 'filename'):
            results.append(f'Filename: {loaded_emb.filename}')

    if isinstance(emb_vec, tuple):
        results.append('SDXL Model Detected')
        for i, vec_name in enumerate(['CLIP-L', 'CLIP-G']):
            vec = emb_vec[i].to(device='cpu', dtype=torch.float32)
            results.append(f'{vec_name} Vector count: {vec.shape[0]}')
            results.append(f'{vec_name} Vector size: {vec.shape[1]}')
    else:
        emb_vec = emb_vec.to(device='cpu', dtype=torch.float32)
        results.append(f'Vector count: {emb_vec.shape[0]}')
        results.append(f'Vector size: {emb_vec.shape[1]}')
    results.append(SEP_STR)

    tokenizer, internal_embs, loaded_embs = get_data()

    if isinstance(internal_embs, tuple):
        all_embs_l = internal_embs[0].to(device='cpu', dtype=torch.float32)
        all_embs_g = internal_embs[1].to(device='cpu', dtype=torch.float32)
    else:
        all_embs = internal_embs.to(device='cpu', dtype=torch.float32)

    torch.set_printoptions(threshold=VEC_SHOW_THRESHOLD, profile=VEC_SHOW_PROFILE)

    if isinstance(emb_vec, tuple):
        for idx, (vec, name) in enumerate(zip(emb_vec, ['CLIP-L', 'CLIP-G'])):
            results.extend(process_vector(vec, name, all_embs_l if idx == 0 else all_embs_g, tokenizer))
    else:
        results.extend(process_vector(emb_vec, 'Vector', all_embs, tokenizer))

    return '\n'.join(results)

def process_vector(emb_vec, name, all_embs, tokenizer):
    results = []
    for v in range(emb_vec.shape[0]):
        vec_v = emb_vec[v].to(device='cpu', dtype=torch.float32)

        results.append(f'{name}[{v}] = {str(vec_v)}')
        results.append(f'Magnitude: {torch.linalg.norm(vec_v).item()}')
        results.append(f'Min, Max: {torch.min(vec_v).item()}, {torch.max(vec_v).item()}')

        if vec_v.shape[0] != all_embs.shape[1]:
            results.append('Vector size is not compatible with current SD model')
            continue

        results.append('')
        results.append("Similar tokens:")
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        scores = cos(all_embs, vec_v.unsqueeze(0))
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        best_ids = sorted_ids[0:MAX_SIMILAR_EMBS].detach().numpy()
        r = []
        for i in range(0, MAX_SIMILAR_EMBS):
            emb_id = best_ids[i].item()
            emb_name = emb_id_to_name(emb_id, tokenizer)

            score_str = ''
            if SHOW_SIMILARITY_SCORE:
                score_str = f' {score_to_percent(sorted_scores[i].item())}% '

            r.append(f'{emb_name}({emb_id}){score_str}')
        results.append('   '.join(r))

        results.append(SEP_STR)
    return results

def parse_mix_inputs_and_sliders(*mix_inputs_and_sliders):
    mix_inputs = list(mix_inputs_and_sliders[:MAX_NUM_MIX])
    mix_sliders = list(mix_inputs_and_sliders[MAX_NUM_MIX:MAX_NUM_MIX*2])
    return mix_inputs, mix_sliders

def create_tot_vec_from_inputs(*mix_inputs_and_sliders, concat_mode, group_handling):
    mix_inputs, mix_sliders = parse_mix_inputs_and_sliders(*mix_inputs_and_sliders)

    tot_vec_l = None
    tot_vec_g = None
    vec_size_l = None
    vec_size_g = None
    log_messages = []

    def process_group(group_vecs):
        nonlocal group_handling
        stacked_vecs = torch.stack(group_vecs)

        if group_handling == "Average":
            return torch.mean(stacked_vecs, dim=0)
        elif group_handling == "Absolute Max Pooling":
            abs_stacked_vecs = torch.abs(stacked_vecs)
            max_indices = torch.argmax(abs_stacked_vecs, dim=0).unsqueeze(0)
            return stacked_vecs.gather(0, max_indices).squeeze(0)
        elif group_handling == "Normed Absolute Max Pooling":
            abs_stacked_vecs = torch.abs(torch.nn.functional.normalize(stacked_vecs, dim=1))
            max_indices = torch.argmax(abs_stacked_vecs, dim=0).unsqueeze(0)
            return stacked_vecs.gather(0, max_indices).squeeze(0)
        elif group_handling == "First Principal Component":
            mean_vec = torch.mean(stacked_vecs, dim=0)
            centered = stacked_vecs - mean_vec
            cov = centered @ centered.T / (stacked_vecs.shape[0] - 1.0)
            _, eigenvectors = torch.linalg.eigh(cov)
            return (eigenvectors @ centered)[-1] + mean_vec
        elif group_handling == "Hyperspherical Centroid":
            normalized_vecs = torch.nn.functional.normalize(stacked_vecs, dim=1)
            mean_vec = torch.mean(normalized_vecs, dim=0)
            return torch.nn.functional.normalize(mean_vec, dim=0)
        elif group_handling == "Softmax Attention":
            attention_weights = torch.softmax(stacked_vecs @ stacked_vecs.T, dim=1)
            return (attention_weights @ stacked_vecs).mean(dim=0)
        else:
            log_messages.append(f"! Unknown group handling method: {group_handling}")
            return None

    for k in range(MAX_NUM_MIX):
        tokenname = mix_inputs[k]
        mixval = mix_sliders[k]
        if (tokenname == '') or (mixval == 0):
            continue

        names = [name.strip().lower() for name in tokenname.split(',')]
        group_vecs_l = []
        group_vecs_g = []
        for name in names:
            emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(name)
            if isinstance(emb_vec, tuple):
                group_vecs_l.append(emb_vec[0].to(device='cpu', dtype=torch.float32))
                group_vecs_g.append(emb_vec[1].to(device='cpu', dtype=torch.float32))
            else:
                group_vecs_l.append(emb_vec.to(device='cpu', dtype=torch.float32))

        if len(group_vecs_l) > 1:
            mix_vec_l = process_group(group_vecs_l)
        else:
            mix_vec_l = group_vecs_l[0]

        if group_vecs_g:
            if len(group_vecs_g) > 1:
                mix_vec_g = process_group(group_vecs_g)
            else:
                mix_vec_g = group_vecs_g[0]

        if vec_size_l is None:
            vec_size_l = mix_vec_l.shape[0]
        else:
            if vec_size_l != mix_vec_l.shape[0]:
                log_messages.append(f'! Vector size is not compatible for CLIP-L, skipping {emb_name}({emb_id})')
                continue

        if not concat_mode:
            if tot_vec_l is None:
                tot_vec_l = torch.zeros_like(mix_vec_l)
            tot_vec_l += mix_vec_l * mixval
        else:
            if tot_vec_l is None:
                tot_vec_l = mix_vec_l * mixval
            else:
                tot_vec_l = torch.cat([tot_vec_l, mix_vec_l * mixval])

        if group_vecs_g:
            if vec_size_g is None:
                vec_size_g = mix_vec_g.shape[0]
            else:
                if vec_size_g != mix_vec_g.shape[0]:
                    log_messages.append(f'! Vector size is not compatible for CLIP-G, skipping {emb_name}({emb_id})')
                    continue

            if not concat_mode:
                if tot_vec_g is None:
                    tot_vec_g = torch.zeros_like(mix_vec_g)
                tot_vec_g += mix_vec_g * mixval
            else:
                if tot_vec_g is None:
                    tot_vec_g = mix_vec_g * mixval
                else:
                    tot_vec_g = torch.cat([tot_vec_g, mix_vec_g * mixval])

        log_messages.append(f'{"+" if not concat_mode else ">"} {emb_name}({emb_id}) x {mixval}')

    if tot_vec_g is not None:
        return (tot_vec_l, tot_vec_g), log_messages
    else:
        return tot_vec_l, log_messages

def do_precompute(*mix_inputs_and_sliders, concat_mode, precompute_expressions, group_handling):
    tot_vec, _ = create_tot_vec_from_inputs(*mix_inputs_and_sliders, concat_mode=concat_mode, group_handling=group_handling)
    precomputed = {}
    precompute_results = []

    def process_slice(slice_vec, slice_name):
        slice_precomputed = {}
        for expr in precompute_expressions.split('\n'):
            if '=' in expr:
                var_names, computation = expr.split('=', 1)
                var_names = [name.strip() for name in var_names.split(',')]
                computation = computation.strip()
                try:
                    result = eval(computation, {
                        'torch': torch,
                        'tot_vec': slice_vec,
                        'maxn': slice_vec.shape[0],
                        **slice_precomputed
                    })
                    if len(var_names) == 1:
                        slice_precomputed[var_names[0]] = result
                        precompute_results.append(f"{slice_name} {var_names[0]}: shape {result.shape}")
                    else:
                        for i, name in enumerate(var_names):
                            slice_precomputed[name] = result[i]
                            precompute_results.append(f"{slice_name} {name}: shape {result[i].shape}")
                except Exception as e:
                    precompute_results.append(f"Error in {slice_name} {', '.join(var_names)}: {str(e)}")
        return slice_precomputed

    if isinstance(tot_vec, tuple):
        tot_vec_l, tot_vec_g = tot_vec
        precomputed['CLIP-L'] = process_slice(tot_vec_l, 'CLIP-L')
        precomputed['CLIP-G'] = process_slice(tot_vec_g, 'CLIP-G')
    else:
        precomputed = process_slice(tot_vec, 'SD1.5')

    return precomputed, '\n'.join(precompute_results)

def on_precompute_click(*args):
    *mix_inputs_and_sliders, concat_mode, precompute_expressions, group_handling = args
    _, precompute_output = do_precompute(*mix_inputs_and_sliders, concat_mode=concat_mode, precompute_expressions=precompute_expressions, group_handling=group_handling)
    return precompute_output

def do_save(*args):
    *mix_inputs_and_sliders, precompute_box, slice_mode, batch_presets, combine_mode, eval_txt, concat_mode, save_name, enable_overwrite, step_text, group_handling = args
    if save_name == '': return 'Filename is empty', None

    results = []

    preset_count = 1
    if batch_presets == True: preset_count = len(EVAL_PRESETS) // 2

    anything_saved = False
    saved_graph = None

    for preset_no in range(preset_count):
        preset_name = ''
        if (preset_no > 0):
            preset_name = '_' + EVAL_PRESETS[preset_no * 2]
            eval_txt = EVAL_PRESETS[preset_no * 2 + 1]

        save_filename = os.path.join(cmd_opts.embeddings_dir, save_name + preset_name + '.pt')
        file_exists = os.path.exists(save_filename)
        if (file_exists):
            if not (enable_overwrite):
                return ('File already exists (' + save_filename + ') overwrite not enabled, aborting.', None)
            else:
                results.append('File already exists, overwrite is enabled')

        step_val = None
        try:
            step_val = int(step_text)
        except:
            step_val = None
            if (step_text != ''): results.append('Step value is invalid, ignoring')

        tot_vec, log_messages = create_tot_vec_from_inputs(*mix_inputs_and_sliders, concat_mode=concat_mode, group_handling=group_handling)
        results.extend(log_messages)

        precomputed, precompute_output = do_precompute(*mix_inputs_and_sliders, concat_mode=concat_mode, precompute_expressions=precompute_box, group_handling=group_handling)
        results.append("Pre-computation results:")
        results.append(precompute_output)

        if tot_vec is None:
            results.append('No embeddings were mixed, nothing to save')
        else:
            if isinstance(tot_vec, tuple):
                tot_vec_l, tot_vec_g = tot_vec
                vectors = [tot_vec_l, tot_vec_g]
            else:
                vectors = [tot_vec]

            for idx, vec in enumerate(vectors):
                if eval_txt != '':
                    try:
                        maxn = vec.shape[0]
                        maxi = vec.shape[1]
                        for n in range(maxn):
                            eval_context = {
                                'torch': torch,
                                'tot_vec': vec,
                                'vec': vec,
                                'maxn': maxn,
                                'maxi': maxi,
                                'vec_mag': torch.linalg.norm(vec[n]),
                                'vec_min': torch.min(vec[n]),
                                'vec_max': torch.max(vec[n]),
                                'n': n,
                                **(precomputed['CLIP-L'] if idx == 0 and isinstance(tot_vec, tuple) else
                                precomputed['CLIP-G'] if idx == 1 and isinstance(tot_vec, tuple) else
                                precomputed)
                            }

                            if eval_txt.startswith('='):
                                for i in range(maxi):
                                    eval_context['v'] = vec[n,i]
                                    eval_context['i'] = i
                                    eval_context['vec'] = vec
                                    ve = eval(eval_txt[1:], eval_context)
                                    vec[n,i] = ve
                            else:
                                eval_context['v'] = vec[n]
                                eval_context['vec'] = vec
                                ve = eval(eval_txt, eval_context)
                                vec[n] = ve
                        results.append(f'Applied eval to {"CLIP-L" if idx == 0 and isinstance(tot_vec, tuple) else "CLIP-G" if idx == 1 else "SD1.5"}: "{eval_txt}"')
                    except Exception as e:
                        results.append(f'🛑 Error evaluating: "{eval_txt}" - {str(e)}')

                if combine_mode and vec.shape[0] > 1:
                    results.append(f'combining {vec.shape[0]} vectors as 1-vector')
                    vec = torch.sum(vec, dim=0, keepdim=True)

                results.append(f'Final embedding size: {vec.shape[0]} x {vec.shape[1]}')

                if vec.shape[0] > 75 and not (slice_mode or combine_mode):
                    results.append('⚠️WARNING: vector count>75, it may not work 🛑')

            final_vec = vectors[0] if len(vectors) == 1 else tuple(vectors)

            if slice_mode:
                num_vectors = final_vec[0].shape[0] if isinstance(final_vec, tuple) else final_vec.shape[0]
                num_digits = max(2, math.ceil(math.log10(num_vectors + 1)))
                for i in range(num_vectors):
                    slice_name = f"{save_name}{i+1:0{num_digits}d}"
                    slice_filename = os.path.join(cmd_opts.embeddings_dir, slice_name + '.pt')
                    if isinstance(final_vec, tuple):
                        slice_vec = {
                            "clip_l": final_vec[0][i],
                            "clip_g": final_vec[1][i]
                        }
                    else:
                        slice_vec = final_vec[i]

                    success, message = save_embedding(slice_vec, slice_filename, step_val)
                    results.append(message)
                    if success:
                        anything_saved = True
                    else:
                        results.append(f'🛑 Error saving slice "{slice_filename}": {message}')
            else:
                if isinstance(final_vec, tuple):
                    final_vec = {
                        "clip_l": final_vec[0],
                        "clip_g": final_vec[1]
                    }

                success, message = save_embedding(final_vec, save_filename, step_val)
                results.append(message)
                if success:
                    if step_val is not None:
                        results.append(f'Setting step value to {step_val}')
                    anything_saved = True
                else:
                    results.append(f'🛑 Error saving "{save_filename}": {message}')

    if anything_saved:
        results.append('Reloading all embeddings')
        try:
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        except:
            sd_hijack.model_hijack.embedding_db.dir_mtime = 0
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return '\n'.join(results)

def save_embedding(vec, filename, step_val=None):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_data = {}

    if isinstance(vec, dict):
        save_data['clip_l'] = vec['clip_l'].unsqueeze(0).cpu() if vec['clip_l'].dim() == 1 else vec['clip_l'].cpu()
        save_data['clip_g'] = vec['clip_g'].unsqueeze(0).cpu() if vec['clip_g'].dim() == 1 else vec['clip_g'].cpu()
    else:
        save_data['string_to_param'] = {'*': vec.unsqueeze(0).cpu() if vec.dim() == 1 else vec.cpu()}

    try:
        torch.save(save_data, filename)
        return True, f'Saved "{filename}" successfully.'
    except Exception as e:
        return False, f'Error saving "{filename}": {str(e)}'

def do_listloaded():
    tokenizer, internal_embs, loaded_embs = get_data()

    results = []
    results.append(f'Loaded embeddings ({len(loaded_embs)}):')
    results.append('')

    for key in loaded_embs.keys():
        try:
            emb = loaded_embs.get(key)

            r = [str(emb.name)]
            if ENABLE_SHOW_CHECKSUM:
                r.append(f' [{emb.checksum()}]')

            if isinstance(emb.vec, tuple):
                r.append(f' CLIP-L Vectors: {emb.vec[0].shape[0]} x {emb.vec[0].shape[1]}')
                r.append(f' CLIP-G Vectors: {emb.vec[1].shape[0]} x {emb.vec[1].shape[1]}')
            else:
                r.append(f' Vectors: {emb.vec.shape[0]} x {emb.vec.shape[1]}')

            if emb.sd_checkpoint_name is not None:
                r.append(f' Ckpt:{emb.sd_checkpoint_name}')

            results.append('    '.join(r))

        except Exception as e:
            results.append(f'🛑 !error! {str(e)}')
            continue

    return '\n'.join(results)

def do_minitokenize(*args):
    mini_input = args[-1].strip().lower()
    mini_sendtomix = args[-2]
    concat_mode = args[-3]
    combine_mode = args[-4]
    mix_inputs = args[0:MAX_NUM_MIX]

    tokenizer, internal_embs, loaded_embs = get_data()

    results = []
    mix_inputs_list = list(mix_inputs)

    groups = re.findall(r'<<.*?>>|[^<>\s]+', mini_input)

    all_ids = []
    for group in groups:
        if group.startswith('<<') and group.endswith('>>'):
            group_content = group[2:-2]
            group_ids = text_to_emb_ids(group_content, tokenizer)
            all_ids.append(group_ids)
            group_str = ','.join([f'#{id}' for id in group_ids])
            group_tokens = [emb_id_to_name(id, tokenizer) for id in group_ids]
            results.append(f"<<{' '.join(group_tokens)}>> {group_str}  ")
        else:
            ids = text_to_emb_ids(group, tokenizer)
            all_ids.extend([[id] for id in ids])
            for id in ids:
                idstr = f'#{id}'
                embstr = emb_id_to_name(id, tokenizer)
                results.append(f"{embstr} {idstr}  ")

    if mini_sendtomix:
        concat_mode = True
        for i, ids in enumerate(all_ids):
            if i < MAX_NUM_MIX:
                mix_inputs_list[i] = ','.join([f'#{id}' for id in ids])
            else:
                break
        for i in range(len(all_ids), MAX_NUM_MIX):
            mix_inputs_list[i] = ''

    combine_mode = False

    return *mix_inputs_list, concat_mode, combine_mode, ' '.join(results)

def do_reset(*args):
    mix_inputs_list = [''] * MAX_NUM_MIX
    mix_slider_list = [1.0] * MAX_NUM_MIX

    return *mix_inputs_list, *mix_slider_list

def do_eval_preset(*args):
    preset_name = args[0]

    result = ''
    for e in range(len(EVAL_PRESETS)//2):
        if preset_name == EVAL_PRESETS[e*2]:
            result = EVAL_PRESETS[e*2+1]
            break

    return result

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_embeddings_by_basename(basenames):
    all_embeddings = []
    for basename in basenames.split(','):
        basename = basename.strip()
        embeddings = [f for f in os.listdir(cmd_opts.embeddings_dir) if f.startswith(basename) and (f.endswith(".pt") or f.endswith(".safetensors"))]
        all_embeddings.extend(embeddings)
    return sorted(all_embeddings, key=natural_sort_key)

def generate_embedding_grid(prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed, batch_size, basenames):
    embedding_files = get_embeddings_by_basename(basenames)
    embedding_names = [os.path.splitext(emb)[0] for emb in embedding_files] # Remove .pt extension

    total_embeddings = len(embedding_names)
    grid_size = math.ceil(math.sqrt(total_embeddings))

    def cell(x, y):
        p.prompt = f"{prompt} {x}"
        return process_images(p)

    def draw_xy_grid(xs, ys, x_label, y_label, cell):
        res = []

        ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
        hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

        for y in range(len(ys)):
            for x in range(len(xs)):
                embedding_index = y * len(xs) + x
                if embedding_index < len(embedding_names):
                    current_embedding = embedding_names[embedding_index]
                    processed = cell(current_embedding, y)
                    res.append(processed.images[0])
                else:
                    res.append(Image.new('RGB', (width, height), color = 'white'))

        grid = images.image_grid(res, rows=len(ys))
        grid = images.draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

        return grid

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=-1,
        seed_resize_from_w=-1,
        seed_enable_extras=False,
        sampler_name=sampler_name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=0,
    )

    xs = embedding_names[:grid_size]
    ys = [f"Row_{i+1}" for i in range(math.ceil(total_embeddings / grid_size))]

    grid = draw_xy_grid(
        xs=xs,
        ys=ys,
        x_label=lambda x: x,
        y_label=lambda y: y,
        cell=cell
    )

    grid_info = {
        "total_embeddings": total_embeddings,
        "grid_size": grid_size,
        "embeddings": embedding_names,
        "generation_params": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "sampler": sampler_name,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "batch_size": batch_size
        }
    }

    return grid, grid_info

def add_tab():

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            with gr.Tab("Mixer"):
                with gr.Row():

                    with gr.Column(variant='panel'):
                        text_input = gr.Textbox(label="Inspect", lines=1, placeholder="Enter name of token/embedding or token ID as #nnnnn")
                        with gr.Row():
                            inspect_button = gr.Button(value="Inspect", variant="primary")
                            listloaded_button = gr.Button(value="List loaded embeddings")
                        inspect_result = gr.Textbox(label="Results", lines=15)

                        with gr.Column(variant='panel'):
                            mini_input = gr.Textbox(label="Mini tokenizer", lines=1, placeholder="Enter a short prompt (loaded embeddings or modifiers are not supported)")
                            with gr.Row():
                                mini_tokenize = gr.Button(value="Tokenize", variant="primary")
                                mini_sendtomix = gr.Checkbox(value=False, label="Send IDs to mixer")
                            mini_result = gr.Textbox(label="Tokens", lines=1)

                    with gr.Column(variant='panel'):
                        with gr.Row():
                            reset_button = gr.Button(value="Reset mixer")
                            group_handling = gr.Dropdown(
                                label="Group Handling Method",
                                choices=["Average",
                                         "Absolute Max Pooling",
                                         "Normed Absolute Max Pooling",
                                         "First Principal Component",
                                         "Hyperspherical Centroid",
                                         "Softmax Attention"],
                                value="Average"
                            )

                        mix_inputs = []
                        mix_sliders = []

                        global SHOW_NUM_MIX
                        if SHOW_NUM_MIX>MAX_NUM_MIX: SHOW_NUM_MIX=MAX_NUM_MIX

                        for n in range(SHOW_NUM_MIX):
                            with gr.Row():
                                with gr.Column():
                                    mix_inputs.append(gr.Textbox(label="Name "+str(n), lines=1, placeholder="Enter name of token/embedding or ID"))
                                with gr.Column():
                                    mix_sliders.append(gr.Slider(label="Multiplier",value=1.0,minimum=-1.0, maximum=1.0, step=0.1))
                        if MAX_NUM_MIX>SHOW_NUM_MIX:
                            with gr.Accordion('',open=False):
                                for n in range(SHOW_NUM_MIX,MAX_NUM_MIX):
                                    with gr.Row():
                                        with gr.Column():
                                            mix_inputs.append(gr.Textbox(label="Name "+str(n), lines=1, placeholder="Enter name of token/embedding or ID"))
                                        with gr.Column():
                                            mix_sliders.append(gr.Slider(label="Multiplier",value=1.0,minimum=-1.0, maximum=1.0, step=0.1))

                        with gr.Row():
                            with gr.Column():
                                precompute_button = gr.Button(value="Pre-compute expressions")
                                precompute_output = gr.Textbox(label="Pre-computed results", lines=3)
                            with gr.Column():
                                precompute_box = gr.Textbox(label="Pre-compute expressions", lines=5, placeholder="Enter expressions to pre-compute, one per line")

                        with gr.Row():
                                with gr.Column():
                                    concat_mode = gr.Checkbox(value=False,label="Concat mode")
                                    combine_mode =  gr.Checkbox(value=False,label="combine as 1-vector")
                                    slice_mode = gr.Checkbox(value=False,label="save 1-vector slices")
                                    step_box = gr.Textbox(label="Step",lines=1,placeholder='only for training')

                                with gr.Column():
                                    preset_names = []
                                    for i in range(len(EVAL_PRESETS)//2):
                                        preset_names.append(EVAL_PRESETS[i*2])
                                    presets_dropdown = gr.Dropdown(label="Eval Preset",choices=preset_names)
                                    eval_box =  gr.Textbox(label="Eval",lines=2,placeholder='')

                        with gr.Row():
                            save_name = gr.Textbox(label="Filename",lines=1,placeholder='Enter file name to save')
                            save_button = gr.Button(value="Save mixed", variant="primary")
                            batch_presets = gr.Checkbox(value=False,label="Save for ALL presets")
                            enable_overwrite = gr.Checkbox(value=False,label="Enable overwrite")

                        with gr.Row():
                            save_result = gr.Textbox(label="Log", lines=10)

            with gr.Tab("Embedding Grid"):
                gr.Markdown("## Generate Embedding Grid")
                with gr.Row():
                    grid_prompt = gr.Textbox(label="Prompt")
                    grid_negative_prompt = gr.Textbox(label="Negative Prompt")
                with gr.Row():
                    grid_steps = gr.Slider(label="Steps", minimum=1, maximum=150, value=20, step=1)
                    grid_sampler = gr.Dropdown(label="Sampling Method", choices=[x.name for x in sd_samplers.samplers], value=sd_samplers.samplers[0].name)
                    grid_cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, value=7, step=0.5)
                with gr.Row():
                    grid_width = gr.Slider(label="Width", minimum=64, maximum=2048, value=512, step=64)
                    grid_height = gr.Slider(label="Height", minimum=64, maximum=2048, value=512, step=64)
                with gr.Row():
                    grid_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    grid_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                grid_basenames = gr.Textbox(label="Embedding Basenames (comma-separated)", value="mixemb_unmixed,mixemb_remixed")
                grid_generate_button = gr.Button("Generate Grid")
                grid_output_image = gr.Image(label="Generated Grid")
                grid_output_info = gr.JSON(label="Grid Info")

            listloaded_button.click(fn=do_listloaded, outputs=inspect_result)
            inspect_button.click(fn=do_inspect,inputs=[text_input],outputs=[inspect_result])
            save_button.click(
                fn=do_save,
                inputs=[*mix_inputs,
                        *mix_sliders,
                        precompute_box,
                        slice_mode,
                        batch_presets,
                        combine_mode,
                        eval_box,
                        concat_mode,
                        save_name,
                        enable_overwrite,
                        step_box,
                        group_handling],
                outputs=[save_result]
            )
            mini_tokenize.click(
                fn=do_minitokenize,
                inputs=mix_inputs+[combine_mode, concat_mode, mini_sendtomix, mini_input],
                outputs=mix_inputs+[concat_mode,combine_mode, mini_result]
            )
            reset_button.click(fn=do_reset,outputs=mix_inputs+mix_sliders)
            precompute_button.click(
                fn=on_precompute_click,
                inputs=[*mix_inputs, *mix_sliders, concat_mode, precompute_box, group_handling],
                outputs=[precompute_output]
            )
            presets_dropdown.change(do_eval_preset,inputs=presets_dropdown,outputs=eval_box)

            grid_generate_button.click(
                fn=generate_embedding_grid,
                inputs=[grid_prompt,
                        grid_negative_prompt,
                        grid_steps,
                        grid_sampler,
                        grid_cfg_scale,
                        grid_width,
                        grid_height,
                        grid_seed,
                        grid_batch_size,
                        grid_basenames],
                outputs=[grid_output_image, grid_output_info]
            )

    return [(ui, "Embedding Remixer", "iremixer")]

script_callbacks.on_ui_tabs(add_tab)
