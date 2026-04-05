#!/bin/bash

# ============================================================================
# OmniVoice TTS – CodeProject.AI module installer (Linux / macOS)
# ============================================================================
# This script is called from the CodeProject.AI setup infrastructure with the
# single argument "install".  It must NOT be run directly.
# ============================================================================

if [ "$1" != "install" ]; then
    echo "This script is only called from: bash ../../src/setup.sh"
    exit 1
fi

# ----------------------------------------------------------------------------
# Platform guards
# ----------------------------------------------------------------------------

if [ "${edgeDevice}" = "Raspberry Pi" ] || [ "${edgeDevice}" = "Jetson" ] || \
   [ "${edgeDevice}" = "Orange Pi" ]; then
    moduleInstallErrors="OmniVoice TTS is not supported on edge devices (${edgeDevice})."
fi

# ----------------------------------------------------------------------------
# System-level audio libraries (Linux only)
# ----------------------------------------------------------------------------

if [ "$moduleInstallErrors" = "" ] && [ "$os" = "linux" ]; then
    installAptPackages "libsndfile1 ffmpeg"
fi

# ----------------------------------------------------------------------------
# Python packages
# ----------------------------------------------------------------------------

if [ "$moduleInstallErrors" = "" ]; then
    installPythonPackagesByName "codeproject-ai-sdk" "CodeProject.AI SDK"
fi

if [ "$moduleInstallErrors" = "" ]; then
    if [ "$hasCUDA" = true ]; then
        writeLine "Installing PyTorch with CUDA support …" "$color_info"
        installPythonPackagesByName \
            "torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128" \
            "PyTorch (CUDA)"
    elif [ "$os" = "macos" ] && [ "$architecture" = "arm64" ]; then
        writeLine "Installing PyTorch for Apple Silicon (MPS) …" "$color_info"
        installPythonPackagesByName "torch==2.8.0 torchaudio==2.8.0" "PyTorch (MPS)"
    else
        writeLine "Installing PyTorch (CPU) …" "$color_info"
        installPythonPackagesByName "torch torchaudio" "PyTorch (CPU)"
    fi
fi

if [ "$moduleInstallErrors" = "" ]; then
    installPythonPackagesByName "omnivoice" "OmniVoice"
fi
