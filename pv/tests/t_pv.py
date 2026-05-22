#!/usr/bin/env python3

import sys

import pv

sys.argv = [
    'pv',
    '--cfg', './config.yaml',
    '--plume_data', './data/previous_manual_annotation_oneback.json',
    '--plume_ids', 'CH4_PlumeComplex-3727',
    '--log', 'DEBUG'
    ]

pv.pv.main()
