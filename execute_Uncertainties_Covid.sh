#!/bin/bash
#############################################
# Executes the Uncertainties from end to end#
#############################################
echo "Executing Uncertainty Autoencoders"
if python3 -u Uncertainty_Covid_AE.py	
then 
	echo "Autoencoders trained, executing XGBoost"
	python3 -u Uncertainty_XGBoost.py
	echo "----------------Done--------------------"
fi

