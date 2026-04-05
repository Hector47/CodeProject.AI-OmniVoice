@if "%1" NEQ "install" (
    echo This script is only called from ..\..\src\setup.bat
    @pause
    @goto:eof
)

:: ============================================================================
:: OmniVoice TTS – CodeProject.AI module installer (Windows)
:: ============================================================================

:: Install CodeProject.AI SDK
call "%utilsScript%" InstallPythonPackagesByName "codeproject-ai-sdk" "CodeProject.AI SDK"
if errorlevel 1 goto:eof

:: Install PyTorch
if /i "%hasCUDA%" == "true" (
    echo Installing PyTorch with CUDA support ...
    call "%utilsScript%" InstallPythonPackagesByName "torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128" "PyTorch (CUDA)"
) else (
    echo Installing PyTorch (CPU) ...
    call "%utilsScript%" InstallPythonPackagesByName "torch torchaudio" "PyTorch (CPU)"
)
if errorlevel 1 goto:eof

:: Install OmniVoice
call "%utilsScript%" InstallPythonPackagesByName "omnivoice" "OmniVoice"
if errorlevel 1 goto:eof
