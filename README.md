# spherical-harmonic-power
# Spectral and Correlation Analysis of Spherical Harmonic Coefficients of Points on a Sphere

This repository contains two Python scripts for spectral analysis and correlation analysis of spherical harmonic coefficients. 
These scripts are useful for analyzing spatial data on a sphere.

## Scripts

### `spectral_power.py`

The `spectral_power` script computes the spectral power and spherical harmonic coefficients up to a specified degree and order for given latitudes and longitudes. 
It takes latitude and longitude data as inputs and provides spectral power and coefficient outputs. The results can be saved as CSV files if desired.

### `power_corr.py`

The `power_corr` script calculates the correlation coefficients per degree between two sets of spherical harmonic coefficients. 
It takes cosine and sine coefficients of two datasets as inputs and returns correlation coefficients per degree. 
Confidence intervals can also be provided for the correlation coefficients.

## Usage

1. Ensure you have the necessary dependencies installed: numpy, pandas, scipy, and pyshtools.
2. Run the scripts using Python 3, passing the required inputs as described in the function documentation.
3. Optionally, save the outputs as CSV files by specifying the output filenames.

Please refer to the function documentation within each script for detailed information on input parameters and outputs.

## Examples

Example usage and sample datasets can be found in the `examples` directory.

## License

Feel free to use and modify the scripts according to your needs.

For any questions or issues, please contact Wes Tucker (wtucke5@uic.edu).
