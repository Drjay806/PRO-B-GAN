#!/usr/bin/env python3
"""
Prot-B-GAN Installation Script
=============================

This script installs all required dependencies for Prot-B-GAN with proper version compatibility.
Supports both Google Colab (CUDA 11.8) and local environments.

Usage:
    # In Google Colab:
    !python install_prot_b_gan.py --colab
    
    # Local installation:
    python install_prot_b_gan.py --local
    
    # Check installation:
    python install_prot_b_gan.py --check
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f" {description}")
    print(f"   Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def install_colab():
    """Install for Google Colab with CUDA 11.8"""
    print(" Installing Prot-B-GAN dependencies for Google Colab (CUDA 11.8)...")
    
    commands = [
        ('pip install "numpy<2.0"', "Installing NumPy < 2.0"),
        ('pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', 
         "Installing PyTorch 2.0.0 with CUDA 11.8"),
        ('pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html',
         "Installing PyTorch Geometric dependencies"),
        ('pip install torch-geometric', "Installing PyTorch Geometric"),
        ('pip install scikit-learn pandas matplotlib tqdm', "Installing additional dependencies")
    ]
    
    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False
    
    if success:
        print("\n Colab installation completed successfully!")
        print(" You can now run: !python prot_b_gan.py --data_root /path/to/data --debug --verbose")
    else:
        print("\n Some installations failed. Please check the error messages above.")
    
    return success

def install_local():
    """Install for local environment"""
    print(" Installing Prot-B-GAN dependencies for local environment...")
    
    commands = [
        ('pip install "numpy<2.0"', "Installing NumPy < 2.0"),
        ('pip install torch==2.0.0 torchvision torchaudio', "Installing PyTorch 2.0.0"),
        ('pip install torch-geometric', "Installing PyTorch Geometric"),
        ('pip install scikit-learn pandas matplotlib tqdm', "Installing additional dependencies")
    ]
    
    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False
    
    if success:
        print("\n Local installation completed successfully!")
        print(" You can now run: python prot_b_gan.py --data_root /path/to/data --debug --verbose")
    else:
        print("\n Some installations failed. Please check the error messages above.")
    
    return success

def check_installation():
    """Check if installation is working correctly"""
    print("🔍 Checking Prot-B-GAN installation...")
    
    checks = [
        ("import numpy", "NumPy"),
        ("import torch", "PyTorch"),
        ("import torch_geometric", "PyTorch Geometric"),
        ("import sklearn", "Scikit-learn"),
        ("import pandas", "Pandas"),
        ("import matplotlib", "Matplotlib"),
        ("import tqdm", "tqdm")
    ]
    
    success = True
    versions = {}
    
    for import_cmd, name in checks:
        try:
            exec(import_cmd)
            print(f" {name} - OK")
            
            # Get version if possible
            if name == "NumPy":
                import numpy
                versions[name] = numpy.__version__
            elif name == "PyTorch":
                import torch
                versions[name] = torch.__version__
            elif name == "PyTorch Geometric":
                import torch_geometric
                versions[name] = torch_geometric.__version__
            elif name == "Scikit-learn":
                import sklearn
                versions[name] = sklearn.__version__
            elif name == "Pandas":
                import pandas
                versions[name] = pandas.__version__
                
        except ImportError as e:
            print(f" {name} - FAILED: {e}")
            success = False
    
    print(f"\n Version Information:")
    for package, version in versions.items():
        print(f"   {package}: {version}")
    
    # Additional PyTorch checks
    try:
        import torch
        print(f"\n🔧 PyTorch Configuration:")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Check for version compatibility issues
        numpy_version = tuple(map(int, versions.get("NumPy", "0.0").split('.')[:2]))
        if numpy_version >= (2, 0):
            print(f" WARNING: NumPy {versions['NumPy']} detected (≥2.0)")
            print(f"      This may cause compatibility issues with PyTorch")
            success = False
            
    except Exception as e:
        print(f"PyTorch configuration check failed: {e}")
        success = False
    
    if success:
        print(f"\nAll checks passed! Prot-B-GAN is ready to use.")
        print(f"Try running: python prot_b_gan.py --help")
    else:
        print(f"\n Some checks failed. Please reinstall the problematic packages.")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Install Prot-B-GAN dependencies')
    parser.add_argument('--colab', action='store_true', help='Install for Google Colab (CUDA 11.8)')
    parser.add_argument('--local', action='store_true', help='Install for local environment')
    parser.add_argument('--check', action='store_true', help='Check installation')
    
    args = parser.parse_args()
    
    if args.colab:
        return 0 if install_colab() else 1
    elif args.local:
        return 0 if install_local() else 1
    elif args.check:
        return 0 if check_installation() else 1
    else:
        print("Please specify installation target:")
        print("  --colab   Install for Google Colab")
        print("  --local   Install for local environment")
        print("  --check   Check existing installation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
