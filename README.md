# plume-vetting
A repository to vet trace gas plumes using atmospheric transmittance profiles.

The initial commit includes the code and parameterization used in [Xiang, Chuchu and Thompson, David R. and Green, Robert O. and Fahlen, Jay E. and Thorpe, Andrew K. and Brodrick, Philip G. and Coleman, Red Willow and Lopez, Amanda M. and Elder, Clayton D., Identification of False Methane Plumes for Orbital Imaging Spectrometers: A Case Study with Emit.](https://dx.doi.org/10.2139/ssrn.5006644)

# File Overview
1.	loop_plume.py: the main script.
2.	sfun.py: contains supporting functions called by the main script.
3.	emit_data_c.py: generates the input data needed by ghg_process.py.
4.	ghg_process.py and target_generation.py: generate the methane signal using the new lookup table. Each has two versions—old and new (as indicated in the file name).
o	The old version does not account for how the slope of log(radiance) varies with concentration length.
o	The new version accounts for this dependence and provides options to either generate a single signal using the mean atmospheric and observational conditions of the selected pixels, or generate individual signals for each pixel based on their local atmospheric and observational conditions.

# How It Runs
•	In loop_plume.main(), adjustable parameters are noted in a comment section. if loop_all_plumes == 1 

•	The primary computation occurs in loop_plume.process_data(), where users can adjust parameters in the top comment block section. This function calls sfun.calculate_radiance_ratio to compute the target-to-background radiance ratio. Additional parameters for pixel selection can be modified in lines comment block of the get_radiance_ratio function in sfun.py.

•	The adjustable parameters are currently set to the same values as those used in the paper.
