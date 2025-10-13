#!/usr/bin/env bash

#
# download some test data for use in local test validation cases.
# note that 'dataproducts_root' locations are machine/data repo dependent, and
# corresponding careted quantities ("< >") are host/user dependent and thus
# need to be provided in order to run.
#

set -x

# configuration (also, ref. config.yaml):

username=<username>
remote=<remote_host>

emit_acquistion_dataproducts_root=/store/emit/ops/data/acquisitions
methane_ver=methane_20230813
emit_matched_filter_dataproducts_root=/store/brodrick/methane/${methane_ver}

date=20241130
fid=emit20241130t180334

# some useful GeoJSON plume FeatureCollections:
plume_feature_collection='/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json'
#plume_feature_collection='/store/brodrick/methane/decent_snowflake_predicted_plumes.json'

local_data_root='./data'

mkdir ${local_data_root}
mkdir -p ./${local_data_root}/acquisitions/${date}/${fid}/l1b
mkdir -p ./${local_data_root}/acquisitions/${date}/${fid}/l2a
mkdir -p ./${local_data_root}/${methane_ver}/${date}

# plume feature collection:
scp ${username}@${remote}:${plume_feature_collection} ${local_data_root}/

# somewhat randomly-chosen acquisition scene and plume data:
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l1b/*mask*"  ${local_data_root}/acquisitions/${date}/${fid}/l1b/
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l1b/*glt*"   ${local_data_root}/acquisitions/${date}/${fid}/l1b/
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l1b/*loc*"   ${local_data_root}/acquisitions/${date}/${fid}/l1b/
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l1b/*obs*"   ${local_data_root}/acquisitions/${date}/${fid}/l1b/
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l1b/*rdn*"   ${local_data_root}/acquisitions/${date}/${fid}/l1b/
scp "${username}@${remote}:${emit_acquistion_dataproducts_root}/${date}/${fid}/l2a/*mask*"  ${local_data_root}/acquisitions/${date}/${fid}/l2a/

# corresponding matched filter files:
scp "${username}@${remote}:${emit_matched_filter_dataproducts_root}/${date}/${fid}_ch4*"    ${local_data_root}/${methane_ver}/${date}/

