"""
This file contains code for the following paper:
"Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with
Multi-Information Derivative-Free Control"
"""


import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='args parser')

    parser.add_argument('--gpu-id', type=int, default=0, help='Specify the GPU.')
    parser.add_argument('--cfg-path', required=True, help='Path to configuration file.')
    parser.add_argument('--cycle-num', type=int, default=1, help='Number of cycles to run the script.')
    parser.add_argument('--iteration-num', type=int, default=1, help='Number of iterations to run in each cycle.')
    parser.add_argument('--lbl-ld', type=bool, default=False, help='Flag to indicate if the language description is line-by-line.')
    parser.add_argument('--lbl-text-append', type=str, default='Please compare the language description with the scene in the image and answer the question if they match in shape, position, and color (yes or no). If they do not match, add a sentence stating where they mismatch. Does the language description match the image? Answer: [insert yes or no] Language description: {llm_message}', help='Text to append for VLM conversation.')
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset.')
    parser.add_argument('--video-name', type=str, default='', help='Scene name.')
    parser.add_argument('--num_actions', type=int, default=10, help='Number of actions in the sequence.')
    parser.add_argument('--demo_data', type=str, required=True, help='Path to the demo data.')
    parser.add_argument('--scene_attribute_match', type=float, default=0.5, help='Proportion of scene attributes for optimal viewpoints.')
    parser.add_argument('--phase_train_infer', type=str, required=True, help='Specify the code phase: train or infer.')
    parser.add_argument('--cs_level', type=str, default='level_sum', help='Complexity score level.')
    parser.add_argument('--xmr_call', type=str, default='True', help='Toggle for running the code in debugging mode.')
    parser.add_argument('--num_scns_processed', type=int, default=0, help='Number of scenes processed in a run.')
    parser.add_argument('--csm_type', type=str, required=True, help='Specify the complexity score metric type: goledol or ghled or complexity_scores.')
    parser.add_argument('--review_mode', action='store_true', help="Enable review mode for the script.")

    return parser