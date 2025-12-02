"""
This file contains an outline of the code for the following paper:
"Video and Language Alignment in 2D Systems for 3D Multi-object Scenes with
Multi-Information Derivative-Free Control"

Please refer to the Method section in the main paper and the Technical Appendix for details on the controller and MI-ZO algorithm.
"""


# Packages
import os
import json
import shutil
import re
from collections import defaultdict
import random
import argparse
from pathlib import Path
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
# Project files
from is_cam import RunCam
from cmu_mod import DataProcessingCS, ComponentModel1, ComponentModel2, CentralUnit, ILSRSSseq, ILSRSScs


# Set random seed
random.seed(12) 

class VisualReviewProcessor:
    """
    Process responses from VLM.
    """

        return self.json_in_dir
    

class SceneAttributeExtractor:
    """
    Process scene attributes responses from VLM.
    """


class CMU:
    """
    Specify the controller and estimate sequence of actions.
    Please refer to the Method section in the main paper and the Technical Appendix for details.
    """

        return self.current_estimate, updated_historic_camera_data, current_camera_data


class UtilsCMU:
    @staticmethod
    """
    Utils for the controller.
    """


class PolynomialRegressionModel:
    """
    Controller submodule for polynomial regression.
    """


class MIZO:
    """
    MI-ZO algorithm for estimating multi-information on cross-modal inputs with active regret normalisation.
    See the Method section of the main report for algorithm operations and specifications.
    Theoretical analysis is presented in the Technical Outline with a sketch in the Method section. 
    """
    def ZO_optim(poly_model, X1_train, X2_train, y_train, n_iterations=100, eta=0.01):
        """
        Zeroth-order optimisation.
        """


class ProcessXMRResponses:
    """
    Process VLM responses for debugging mode.
    """

        return xmr_responses, xmr_scene_attributes, xmr_scene_attributes_all_responses


class PropertiesData:
    """
    Process metadata on scene properties
    """


class TXTMetricAdjustment:
    """
    Calculate text metrics ie noun phrases and descriptive terms.
    """

        return txt_results


class RunCMU:
    """
    Class to run the controller.
    """

    # Initialise and run controller and MI-ZO
    def run_task(self, viewpoints, current_camera_data, iteration, video_name, actual_labels, dataset_dir):
        """
        Main loop for running the controller and MI-ZO.
        Please refer to the Method and Experiments sections in the main paper. 
        Additional details are provided in the Technical Appendix.
        """
