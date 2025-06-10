# ParManus AI Agent with Ollama Integration

A streamlined AI agent system using Ollama for local LLM inference with Llama 3.2 support.

## Features

- **Ollama Integration**: Native support for Ollama API with Llama 3.2 models
- **Vision Support**: Multi-modal capabilities with vision models
- **Streamlined Architecture**: Single main.py handles all functionality
- **Memory System**: Session persistence and recovery
- **Agent Routing**: Automatic agent selection based on query type
- **Voice Support**: Optional TTS/STT integration
- **Optimized Dependencies**: Minimal required packages

## Quick Start

### Prerequisites

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull Llama 3.2 models:
```bash
ollama pull llama3.2
ollama pull llama3.2-vision  # Optional for vision support
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mrarejimmyz/ParManusAI.git
cd ParManusAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure (optional):
```bash
cp config/config.toml config/config.toml.local
# Edit config/config.toml.local as needed
```

### Usage

#### Basic Usage
```bash
python main.py --prompt "Hello, how are you?"
```

#### Interactive Mode
```bash
python main.py
```

#### Specify Agent
```bash
python main.py --agent code --prompt "Write a Python function to calculate fibonacci"
```

#### With Custom Config
```bash
python main.py --config config/config.toml.local --prompt "Analyze this image"
```

#### Voice Mode (requires voice dependencies)
```bash
python main.py --voice
```

## Configuration

The system uses TOML configuration files. Default configuration:

```toml
[llm]
api_type = 'ollama'
model = "llama3.2"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
max_tokens = 2048
temperature = 0.0

[llm.vision]
api_type = 'ollama'
model = "llama3.2-vision"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
max_tokens = 2048
temperature = 0.0

[browser]
headless = false
disable_security = true
extra_chromium_args = []
```

## Available Agents

- **manus**: General-purpose AI assistant (default)
- **code**: Programming and development tasks
- **browser**: Web automation and scraping
- **file**: File operations and data processing
- **planner**: Task planning and organization

## Architecture

The optimized architecture consists of:

- `main.py`: Single entry point handling all functionality
- `config/`: Configuration files
- Minimal dependencies for core functionality
- Optional modules for extended features

## Migration from Legacy

The system maintains backward compatibility while providing a streamlined experience:

- Legacy llama-cpp-python code moved to `app/llm_legacy.py`
- Original main.py backed up as `main_original.py`
- Original requirements backed up as `requirements_original.txt`

## Development

### Adding New Agents

Extend the `SimpleAgent` class or modify the `route_agent` function in `main.py`.

### Custom LLM Providers

Modify the `OllamaLLM` class or create new implementations following the same interface.

### Configuration Options

Add new configuration fields to the `Config` model in `main.py`.

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if models are available: `ollama list`
- Verify base_url in configuration

### Memory Issues
- Reduce max_tokens in configuration
- Use smaller models if available
- Enable memory compression in config

### Performance Optimization
- Use GPU acceleration with Ollama
- Adjust model parameters in configuration
- Enable session caching

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review Ollama documentation for model-specific issues

