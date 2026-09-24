# pkg_visualizer v0.1.1 Release Notes

**Release Date:** December 7, 2025  
**Author:** Eric G. Suchanek, PhD

## Overview

We're excited to announce the initial release of **pkg_visualizer** v0.1.1, a powerful Python-based application that brings 3D visualization to Python package exploration. This tool transforms static code analysis into an interactive, visual experience, making it easier for developers and researchers to understand complex codebases at a glance.

## 🚀 Key Features

### Interactive 3D Visualization
- **Dynamic object sizing** with runtime-adjustable class radius slider
- **Adaptive rendering** that automatically selects optimal shapes based on package complexity:
  - Icosahedrons for classes (small-medium packages) → Cubes (large packages)
  - Cylinders for functions (≤1000) → Cubes (>1000)
- **Intelligent spatial distribution** using Fibonacci sphere algorithms for optimal element placement
- **Hierarchical organization** with classes, methods, and functions positioned in distinct spatial layers

### Advanced User Interface
- **Comprehensive selection controls** for classes, methods, and functions
- **Smart UI state management** with context-aware enable/disable of selectors
- **Real-time visualization updates** via the Visualize button
- **Intuitive camera controls** including Reset View and animated 360° Spin Package
- **Multi-format export** supporting HTML, PNG, and JPG with preserved camera states

### Interactive Code Exploration
- **Docstring integration** with Markdown-formatted popup displays
- **Precision picking system** supporting both mouse clicks and keyboard shortcuts (P key)
- **Automatic highlighting and zoom** for selected code elements
- **Priority-based selection** (methods take precedence over parent classes)
- **Non-blocking popup windows** for seamless exploration workflow

### Performance & Scalability
- **Adaptive connection rendering**:
  - Detailed cylinders (< 500 classes)
  - Optimized lines (500-2000 classes)  
  - Disabled connections (> 2000 classes)
- **Memory management** with comprehensive cleanup and exception handling
- **Progress tracking** with real-time status updates and triangle count metrics
- **AST-based parsing** for safe code analysis without execution

## 🔧 Technical Highlights

### Architecture
- **PyVista + PyQt5 integration** through QtInteractor for seamless 3D rendering in Qt applications
- **MultiBlock mesh collections** for efficient rendering of complex scenes
- **Robust file I/O** with automatic directory creation and comprehensive error handling
- **Event-driven UI updates** maintaining responsiveness during intensive operations

### Algorithm Innovation
- **Fibonacci sphere distribution** ensures uniform spatial arrangement even with thousands of elements
- **Duplicate detection system** prevents redundant visualization of repeated code elements
- **Hierarchical mesh mapping** enables precise object identification and interaction
- **Adaptive geometry selection** balances visual quality with performance

## 📦 Installation & Usage

### Quick Start
```bash
git clone https://github.com/suchanek/repo_vis
cd repo_vis
poetry install
poetry run pkg-visualizer --package_path /path/to/your/package
```

### Command Line Options
- `--package_path`: Target Python package directory
- `--save_path`: Output file base path (without extension)
- `--width`: Visualization window width (default: 1200)
- `--height`: Visualization window height (default: 800)

## 🎯 Example Visualizations

The release includes pre-rendered HTML visualizations of popular Python projects:
- **Flask** - Web framework architecture
- **Matplotlib** - Plotting library structure  
- **Requests** - HTTP library organization
- **Seaborn** - Statistical visualization components
- **SymPy** - Symbolic mathematics hierarchy
- **TensorFlow** - Machine learning framework layout
- **Scikit-learn** - Machine learning toolkit structure

## 🎮 Interaction Guide

### Navigation
- **Mouse drag**: Rotate camera around the scene
- **Mouse wheel**: Zoom in/out
- **Right-click or P key**: Pick and highlight objects
- **Reset View button**: Return to default camera position
- **Spin Package button**: Animated 360° rotation

### Selection Workflow
1. Use checkboxes to enable/disable methods and functions rendering
2. Select specific elements via dropdown selectors
3. Click **Visualize** to generate/update the 3D scene
4. Interact with objects to explore docstrings and code structure

## 🔍 Under the Hood

### Code Analysis Engine
- **AST-based parsing** safely extracts structural information without code execution
- **Recursive directory traversal** processes entire package hierarchies
- **Intelligent deduplication** ensures clean visualization of complex inheritance patterns

### Rendering Pipeline
- **Adaptive mesh generation** optimizes performance based on package size
- **Real-time triangle counting** provides performance metrics
- **Progressive rendering** with status updates for large codebases
- **Memory-efficient cleanup** prevents resource leaks during application lifecycle

## 🎉 What's Next

This initial release establishes a solid foundation for 3D code visualization. Future versions will expand on these capabilities with additional analysis features, enhanced interactivity, and broader language support.

## 🙏 Acknowledgments

Built with the powerful combination of PyVista for 3D visualization and PyQt5 for user interface, pkg_visualizer demonstrates the potential of visual code analysis tools in modern software development.

---

**Download:** Available on [GitHub](https://github.com/suchanek/repo_vis)  
**Documentation:** See README.md for detailed usage instructions  
**Examples:** Explore the `/examples` directory for interactive demonstrations

Happy visualizing! 🚀
