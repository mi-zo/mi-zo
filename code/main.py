"""
This file contains code for the following paper:
"Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with
Multi-Information Derivative-Free Control"

Please refer to the Method section in the main paper and the supplementary material for additional details.
"""


# Initialize argument parser
# Import args
from utils.args import get_parser
parser = get_parser()
args = parser.parse_args()

# Review mode check
import sys
if args.review_mode:
    print('Review mode called')
    sys.exit('exit')

# Packages
import os
import json
import re
import shutil
import glob
import argparse
import subprocess
import time
import random
import numpy as np
import importlib
import torch
import torch.backends.cudnn as cudnn
# Project files
import cmu_mizo


def initialize_objects(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    # Please refer to the Technical Appendix for GPU specifications
    model_config.device = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

    return chat, model, vis_processor

def get_next_json_num(data_out_dir):
    os.makedirs(data_out_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(data_out_dir) if os.path.isfile(os.path.join(data_out_dir, f))]
    existing_json_files = [f for f in existing_files if f.startswith('combined_data_') and f.endswith('.json')]

    return len(existing_json_files) + 1

def split_ld_text(ld_text):
    instructions = []
    
    sections = ld_text.split('<dsta>')
    for section in sections:
        if '<dend>' in section:
            instruction = section.split('<dend>')[0].strip()
            instructions.append(instruction)
            
    return instructions

def run_video_to_attributes_text_system(args, cycle, iteration, script_name, chat_dir, chat, model, vis_processor, images):
    """
    Conversation to generate attributes.
    See Experiments section in the main report for VLM system settings. 
    """
    attr_text = []
    output_file_path = os.path.join(chat_dir, f'{script_name}_attr_chat_{cycle}_{iteration}.txt')

    def print_and_log(message):
        print(message)
        with open(output_file_path, 'a') as f:
            f.write(message + '\n')

    print_and_log(f'iteration: {iteration}')

    # Process frame
    for round_num, image_file in enumerate(images):
        _, file_extension = os.path.splitext(image_file)
        if file_extension.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f'Unsupported file format: {file_extension}')
            continue

        img_list = []
        chat_state = Conversation(
            system='You are able to understand the visual content that the user provides.',
            roles=('Human', 'Assistant'),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep='###',
        )
        chat.upload_img(image_file, chat_state, img_list)

        input_text = 'In the image, a 3D scene with objects is presented. Please inspect the image and return the following template for each object (ie a separate copy for each): \'For object [insert number] on [insert position of the object], the object has [insert number] corners and [insert number] sides. The nearest side of the object has the color [insert color]. Total number of colors: [enter number].\''
        temperature = 1.0

        chat.ask(input_text, chat_state)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=2,
                                  repetition_penalty=1.1,
                                  temperature=temperature,
                                  max_new_tokens=250,
                                  max_length=2000)[0]

        formatted_input = f'input_{round_num + 1}: <ista> {input_text} <iend>'
        formatted_response = f'response_{round_num + 1}: <rsta> {llm_message} <rend>'

        print_and_log(f'temperature: {temperature}')
        print_and_log(formatted_input)
        print_and_log(formatted_response)

        attr_text.append(formatted_input)
        attr_text.append(formatted_response)

    final_round_num = len(images)

    pattern_to_exclude = r'input_\d+: <ista>.*?For object \[insert number\] on \[insert position of the object\],.*?<iend>'
    total_colors_pattern = r'Total number of colors:.*?(?=<rsta>|\n|$)'
    response_label_pattern = r'response_\d+:'

    combined_current_text = ''

    for response in attr_text:
        response_processed = re.sub(pattern_to_exclude, '', response, flags=re.DOTALL)
        response_processed = re.sub(total_colors_pattern, '', response_processed, flags=re.DOTALL)
        response_processed = re.sub(response_label_pattern, '', response_processed)

        if response_processed not in combined_current_text:
            combined_current_text += ' ' + response_processed

    clean_tokens = ['<ista>', '<iend>', '<rsta>', '<rend>', '</s>', '\\n']
    for token in clean_tokens:
        combined_current_text = combined_current_text.replace(token, '')
    combined_current_text = combined_current_text.replace('_', ' ')
    combined_current_text = combined_current_text.strip()

    final_input_text = f'input_{final_round_num + 1}: <ista> Please read all the inputs and responses: {combined_current_text} Based on these inputs and responses, please complete this sentence: List of colors mentioned: [list colors]. Total number of colors mentioned: [enter number].<iend>'

    chat_state = Conversation(
        system='You are able to understand the textual content that the user provides.',
        roles=('Human', 'Assistant'),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep='###',
    )

    chat.ask(final_input_text, chat_state)
    final_llm_message = chat.answer(conv=chat_state,
                                    img_list=[],
                                    num_beams=2,
                                    repetition_penalty=1.1,
                                    temperature=1.0,
                                    max_new_tokens=250,
                                    max_length=2000)[0]

    final_formatted_response = f'response_{final_round_num + 1}: <rsta> {final_llm_message} <rend>'

    print_and_log(f'temperature: {temperature}')
    print_and_log(final_input_text)
    print_and_log(final_formatted_response)

    attr_text.append(final_input_text)
    attr_text.append(final_formatted_response)

    return attr_text

def run_video_to_text_system(args, cycle, iteration, script_name, chat_dir, chat, model, vis_processor, images, ld_text):
    """
    Conversation on visual and language description inputs.
    See Experiments section in the main report for VLM system settings. 
    """
    modified_ld_text = []
    prepared_ld_text = split_ld_text(ld_text)
    output_file_path = os.path.join(chat_dir, f'{script_name}_chat_{cycle}_{iteration}.txt')

    def print_and_log(message):
        print(message)
        with open(output_file_path, 'a') as f:
            f.write(message + '\n')

    print_and_log(f'iteration: {iteration}')

    # Process image
    for round_num, (image_file, instruction) in enumerate(zip(images, prepared_ld_text[:-1])):
        _, file_extension = os.path.splitext(image_file)
        if file_extension.lower() not in ['.png', '.jpg', '.jpeg']:
            print(f'Unsupported file format: {file_extension}')
            continue

        img_list = []
        chat_state = Conversation(
            system='You are able to understand the visual content that the user provides.',
            roles=('Human', 'Assistant'),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep='###',
        )
        chat.upload_img(image_file, chat_state, img_list)

        input_text = args.lbl_text_append.replace('{instruction}', instruction)
        temperature = 0.1

        chat.ask(input_text, chat_state)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=2,
                                  repetition_penalty=1.1,
                                  temperature=temperature,
                                  max_new_tokens=250,
                                  max_length=2000)[0]

        formatted_input = f'input_{round_num + 1}: <ista> {input_text} <iend>'
        formatted_response = f'response_{round_num + 1}: <rsta> {llm_message} <rend>'

        print_and_log(f'temperature: {temperature}')
        print_and_log(formatted_input)
        print_and_log(formatted_response)

        modified_ld_text.append(formatted_input)
        modified_ld_text.append(formatted_response)

    final_round_num = len(images)
    final_instruction = prepared_ld_text[-1]

    combined_current_text = ' '.join(modified_ld_text)

    lang_desc_response_pattern = r'input_\d+: <ista>.*?Language description: ((?!There).+?) <iend> response_\d+: <rsta>\s*(True|False).*?<rend>'

    matches = re.findall(lang_desc_response_pattern, combined_current_text, flags=re.DOTALL)

    processed_descriptions = []
    for desc, response in matches:
        processed_desc = f'This language description is {response}: {desc}'
        processed_descriptions.append(processed_desc)

    combined_current_text = ' '.join(processed_descriptions) if processed_descriptions else 'No notable features'

    final_input_text = f'input_{final_round_num + 1}: <ista> Please read the description of the 3D scene: {combined_current_text} Based on this description, please answer if this next statement is true or false: {final_instruction} Your answer (pick one of the following and limit your response to one word) [insert True, False] <iend>'

    chat_state = Conversation(
        system='You are able to understand the textual content that the user provides.',
        roles=('Human', 'Assistant'),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep='###',
    )

    chat.ask(final_input_text, chat_state)
    final_llm_message = chat.answer(conv=chat_state,
                                    img_list=[],
                                    num_beams=2,
                                    repetition_penalty=1.1,
                                    temperature=1.0,
                                    max_new_tokens=250,
                                    max_length=2000)[0]

    final_formatted_response = f'response_{final_round_num + 1}: <rsta> {final_llm_message} <rend>'

    print_and_log(f'temperature: {temperature}')
    print_and_log(final_input_text)
    print_and_log(final_formatted_response)

    modified_ld_text.append(final_input_text)
    modified_ld_text.append(final_formatted_response)

    return modified_ld_text

def combine_output_files_attr(output_dir, script_name, chat_dir, current_cycle, iteration_num):
    combined_data = []
    data_out_dir = os.path.join(output_dir, 'data_attr')

    for iteration in range(1, iteration_num + 1):
        output_file_path = os.path.join(chat_dir, f'{script_name}_attr_chat_{current_cycle}_{iteration}.txt')

        if not os.path.exists(output_file_path):
            continue
        
        with open(output_file_path, 'r') as f:
            text = f.read()
            
            match = re.search(r'iteration: (\d+)', text)
            if match:
                file_iteration = int(match.group(1))
            else:
                print(f'[combine_output_files_attr] Error: Iteration file missing in: {output_file_path}')
                continue
            
            conversations = text.split('temperature:')
            chat_num = 0
            for conversation in conversations[1:]:
                chat_num += 1
                lines = conversation.split('\n')
                data = {'cycle': str(current_cycle).zfill(6), 'iteration': str(file_iteration).zfill(6), 'chat_num': chat_num}
                       
                # Extract input text
                input_start_index = None
                input_end_index = None
                for idx, line in enumerate(lines):
                    if '<ista>' in line:
                        input_start_index = idx
                    if '<iend>' in line:
                        input_end_index = idx
                        break
                if input_start_index is not None and input_end_index is not None:
                    data['input'] = '\n'.join(lines[input_start_index:input_end_index+1])
                else:
                    print(f'Error: language description for chat_num {chat_num} missing')
                                                               
                response_start_index = None
                response_end_index = None
                for idx, line in enumerate(lines):
                    if '<rsta>' in line:
                        response_start_index = idx
                    if '<rend>' in line:
                        response_end_index = idx
                        break
                if response_start_index is not None and response_end_index is not None:
                    data['response'] = '\n'.join(lines[response_start_index:response_end_index+1])
                else:
                    print(f'Error: language description for chat_num {chat_num} missing')
                
                combined_data.append(data)
    
    output_json_num = get_next_json_num(data_out_dir)
    file_data_out_dir = os.path.join(data_out_dir, f'combined_data_attr_{output_json_num}.json')
    with open(file_data_out_dir, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    return file_data_out_dir

def combine_output_files(output_dir, script_name, chat_dir, current_cycle, iteration_num):
    combined_data = []
    data_out_dir = os.path.join(output_dir, 'data')
    
    for iteration in range(1, iteration_num + 1):
        output_file_path = os.path.join(chat_dir, f'{script_name}_chat_{current_cycle}_{iteration}.txt')
        
        if not os.path.exists(output_file_path):
            print(f'Error: language description for chat_num  {output_file_path} missing')
            continue
        
        with open(output_file_path, 'r') as f:
            text = f.read()
            
            match = re.search(r'iteration: (\d+)', text)
            if match:
                file_iteration = int(match.group(1))
            else:
                print(f'Error: language description for chat_num  {output_file_path} missing')
                continue
            
            conversations = text.split('temperature:')
            chat_num = 0
            for conversation in conversations[1:]:
                chat_num += 1
                lines = conversation.split('\n')
                data = {'cycle': str(current_cycle).zfill(6), 'iteration': str(file_iteration).zfill(6), 'chat_num': chat_num}
                
                input_start_index = None
                input_end_index = None
                for idx, line in enumerate(lines):
                    if '<ista>' in line:
                        input_start_index = idx
                    if '<iend>' in line:
                        input_end_index = idx
                        break
                if input_start_index is not None and input_end_index is not None:
                    data['input'] = '\n'.join(lines[input_start_index:input_end_index+1])
                else:
                    print(f'Error: language description for chat_num {chat_num} missing')
                                                               
                # Extract response text
                response_start_index = None
                response_end_index = None
                for idx, line in enumerate(lines):
                    if '<rsta>' in line:
                        response_start_index = idx
                    if '<rend>' in line:
                        response_end_index = idx
                        break
                if response_start_index is not None and response_end_index is not None:
                    data['response'] = '\n'.join(lines[response_start_index:response_end_index+1])
                else:
                    print(f'Error: language description for chat_num {chat_num} missing')
                
                combined_data.append(data)
    
    output_json_num = get_next_json_num(data_out_dir)
    file_data_out_dir = os.path.join(data_out_dir, f'combined_data_{output_json_num}.json')
    with open(file_data_out_dir, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    return file_data_out_dir

def process_language_descriptions(ld_text, viewpoints_data):
    ld_dict = json.loads(ld_text) if isinstance(ld_text, str) else ld_text
    formatted_ld = ''
    for label in viewpoints_data:
        if label in ld_dict:
            formatted_ld += ld_dict[label] + ' '
        else:
            print(f'Warning: Label {label} missing from language description.')
            formatted_ld += '<dsta> <dend> '

    to_ld_label = 'to_ld'
    if to_ld_label in ld_dict:
        formatted_ld += ld_dict[to_ld_label]
    else:
        print(f'Warning: summary label missing from language description.')

    return formatted_ld

def load_data_from_json(output_dir_base, video_name):
    viewpoints_dir = os.path.join(output_dir_base, 'viewpoints_camera_data', video_name)
    with open(os.path.join(viewpoints_dir, 'viewpoints.json'), 'r') as vp_file:
        viewpoints = json.load(vp_file)
    with open(os.path.join(viewpoints_dir, 'camera_data.json'), 'r') as cd_file:
        current_camera_data = json.load(cd_file)
    return viewpoints, current_camera_data

def main():

    # Check for debugging
    args.xmr_call = args.xmr_call.lower() in ('true', '1', 't', 'y', 'yes')
    if not args.xmr_call:
        print('Debugging mode.')
    else:
        # Initialise model
        chat, model, vis_processor = initialize_objects(args)

    script_name = os.path.basename(__file__).split('_main')[0]
    output_dir_base = f'path/to/{script_name}'
    os.makedirs(output_dir_base, exist_ok=True)

    json_out_dir = os.path.join(output_dir_base, 'json_out')
    os.makedirs(json_out_dir, exist_ok=True)
    json_out_dir_attr = os.path.join(output_dir_base, 'json_out_attr')
    os.makedirs(json_out_dir_attr, exist_ok=True)

    video_names = args.video_name.split(',')
    args.num_scn_tests = len(video_names)  

    ckpts_all = 'path/to/ckpts'
    ckpts_for_dataset = os.path.join(ckpts_all, args.dataset_name)
    os.makedirs(ckpts_for_dataset, exist_ok=True)

    # Loop through scenes
    for cycle, video_name in enumerate(video_names, start=1):

        args.num_scns_processed += 1 

        chat_dir = os.path.join(output_dir_base, f'chat_dir_{cycle}')
        os.makedirs(chat_dir, exist_ok=True)

        # Loop through iterations
        for iteration in range(1, args.iteration_num + 1):

            results_dir = os.path.join(output_dir_base, 'results', video_name, f'round_{iteration}', 'responses')
            os.makedirs(results_dir, exist_ok=True)
            responses_run_sum_dir = os.path.join(output_dir_base, 'results', 'responses_run_sum_dir')
            os.makedirs(responses_run_sum_dir, exist_ok=True)
            dataset_dir = f'path/to/{args.dataset_name}'
            metadata_file = f'path/to/{args.dataset_name}/metadata'

            # Variables for controller and MI-ZO
            # See the Method section and Technical Appendix of the paper referenced above for details
            metadata_file_path = f'path/to/{args.dataset_name}/metadata'
            metadata_file_name = 'updated_task_3d_cr_md.json'
            metadata_file = os.path.join(metadata_file_path, metadata_file_name)
            metadata_update_path = os.path.join(output_dir_base, 'metadata_update')
            os.makedirs(metadata_update_path, exist_ok=True)
            goledol_file = os.path.join(metadata_file_path, 'goledol_scores.json')
            ghled_file = os.path.join(metadata_file_path, 'ghled_scores.json')

            # Initialise controller and MI-ZO
            # See the Method section and Technical Appendix of the paper referenced above for details
            run_cmu = cmu_module.RunCMU(
                metadata_file = metadata_file,
                num_actions=args.num_actions,
                demo_data=args.demo_data,
                rounds=args.iteration_num,
                output_dir_base=output_dir_base,
                responses_dir=results_dir,
                scene_attribute_match=args.scene_attribute_match,
                num_scn_tests=args.num_scn_tests,
                num_scns_processed = args.num_scns_processed,
                ckpts_for_dataset = ckpts_for_dataset,
                xmr_call=args.xmr_call,
                phase_train_infer=args.phase_train_infer,
                cs_level=args.cs_level,
                csm_type=args.csm_type,
                goledol_file=goledol_file,
                ghled_file=ghled_file
            )

            if iteration == 1:
                run_cmu.default_inputs(video_name)

            viewpoints, current_camera_data = load_data_from_json(output_dir_base, video_name)

            txt_dir_path = f'path/to/{args.dataset_name}/txts/{video_name}'
            txt_files = [f for f in os.listdir(txt_dir_path) if os.path.isfile(os.path.join(txt_dir_path, f))]
            ld_text = ''
            if txt_files:
                txt_file_path = os.path.join(txt_dir_path, txt_files[0])
                print(f'Found txt file for {video_name}: {txt_file_path}')
                with open(txt_file_path, 'r') as file:
                    ld_text = file.read()
            else:
                print(f'txt file for {video_name} missing in {txt_dir_path}')

            viewpoints_zm_levels = [current_camera_data[vp][0] for vp in viewpoints]

            if viewpoints:
                ld_text = process_language_descriptions(ld_text, viewpoints)

            images = []
            for viewpoint in viewpoints:
                zoom_level = current_camera_data[viewpoint][0]
                image_path = f'path/to/{args.dataset_name}/scenes/{video_name}/{zoom_level}/{viewpoint}.png'
                images.append(image_path)
            
            if args.xmr_call:
                attr_text = run_video_to_attributes_text_system(args, cycle, iteration, script_name, chat_dir, chat, model, vis_processor, images)
            else:
                src_attr = 'path/to/default_chat_1_1.txt'
                dest_attr = os.path.join(chat_dir, f'{script_name}_attr_chat_{cycle}_{iteration}.txt')
                shutil.copy(src_attr, dest_attr)

            if not args.xmr_call:
                src_json_path = f'path/to/{video_name}/round_{iteration}/responses/json_process/current_iteration_responses.json'
                dest_json_dir = os.path.join(results_dir, 'json_process')
                os.makedirs(dest_json_dir, exist_ok=True)
                dest_json_path = os.path.join(dest_json_dir, 'current_iteration_responses.json')
                if os.path.normpath(src_json_path) != os.path.normpath(dest_json_path):
                    shutil.copy(src_json_path, dest_json_path)
                else:
                    print(f'Skipping scene for {video_name}, iteration {iteration}.')

            file_data_out_dir_attr = combine_output_files_attr(output_dir_base, script_name, chat_dir, cycle, args.iteration_num)
            file_json_out_dir_attr = os.path.join(json_out_dir_attr, f'combined_data_out_attr_{iteration}.json')
            try:
                shutil.copy(file_data_out_dir_attr, file_json_out_dir_attr)
            except Exception as e:
                print(f'Error occurred while copying attribute file {e}')

            if args.xmr_call:
                run_video_to_text_system(args, cycle, iteration, script_name, chat_dir, chat, model, vis_processor, images, ld_text)
            else:
                src_text = 'path/to/default_chat_1_1.txt'
                dest_text = os.path.join(chat_dir, f'{script_name}_chat_{cycle}_{iteration}.txt')
                shutil.copy(src_text, dest_text)

            # Combine output files for this cycle
            file_data_out_dir = combine_output_files(output_dir_base, script_name, chat_dir, cycle, args.iteration_num)
            file_json_out_dir = os.path.join(json_out_dir, f'combined_data_out_{iteration}.json')
            try:
                shutil.copy(file_data_out_dir, file_json_out_dir)
            except Exception as e:
                print(f'Error occurred while copying combined data file {e}')
            
            # Initialise VisualReviewProcessor
            cmu_module_name = f'{script_name}_cmu'
            cmu_module = importlib.import_module(cmu_module_name)
            video_index = video_names.index(video_name) + 1
            cmu_processor = cmu_module.VisualReviewProcessor(
                video_name=video_name, 
                video_index=video_index, 
                iteration=iteration, 
                metadata_file=metadata_file, 
                metadata_update_path=metadata_update_path,
                responses_dir=results_dir, 
                responses_run_sum_dir=responses_run_sum_dir,
                viewpoints_for_xmr=viewpoints, 
                viewpoints_zm_levels=viewpoints_zm_levels,
                xmr_call=args.xmr_call
            )

            cmu_processor.run_visual_review()

            tf_label_dir = f'path/to/{args.dataset_name}/labels/{video_name}'
            actual_labels_path = glob.glob(f'{tf_label_dir}/*.txt')[0]

            with open(actual_labels_path, 'r') as file:
                actual_labels = json.load(file)

            # Call run_task method to run the controller and MI-ZO
            # See the Method and Experiments sections of the paper referenced above for details
            # Additional information is presented in the Technical Appendix
            run_cmu.run_task(viewpoints=viewpoints, current_camera_data=current_camera_data, iteration=iteration, video_name=video_name, actual_labels=actual_labels, dataset_dir=dataset_dir)

if __name__ == '__main__':
    main()