# AGENTS.md - Customer Chat Assistant

## Project Overview
This is a Flask-based customer analytics application with AI-powered chat assistant using Qwen-Agent. It analyzes customer data, generates visualizations (matplotlib), and provides predictive models (ARIMA, Decision Tree).

## Build & Run Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
# Development mode with auto-reload
python app.py

# Run with debug enabled (default in app.py)
python app.py

# The app will be available at http://127.0.0.1:5000/
```

### Database Configuration
- Modify database connection string in `customer_operation_assistant.py` `ExcSQLTool` class
- Default: MySQL on Aliyun RDS
- Ensure tables `customer_base` and `customer_behavior_assets` exist

### API Key Configuration
- Set `DASHSCOPE_API_KEY` environment variable for Qwen-Agent
- Or modify the default value in `customer_operation_assistant.py`

## Code Style Guidelines

### General Principles
- Write clear, documented code with Chinese comments where appropriate
- Use type hints for function parameters and return values
- Include docstrings for all public functions and classes
- Keep functions focused and under 100 lines when possible

### Imports
```python
# Standard library first, then third-party, then local
import json
import os
import time
from datetime import datetime
from typing import List

import pandas as pd
from flask import Flask, render_template

# Sort imports alphabetically within each group
```

### Naming Conventions
- **Variables/functions**: `snake_case` (e.g., `customer_base_df`, `get_basic_stats`)
- **Classes**: `PascalCase` (e.g., `ExcSQLTool`, `ArimaAUMTool`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DASHSCOPE_API_KEY`)
- **Private members**: Leading underscore (e.g., `_internal_method`)
- **Descriptive names**: Use clear names like `customer_behavior_assets` not `data`

### Type Hints
```python
from typing import List, Optional, Dict

def process_data(df: pd.DataFrame, config: Dict) -> str:
    """Process customer data and return results."""
    result: List[Dict] = []
    return json.dumps(result)

# Use Optional for values that can be None
customer_id: Optional[str] = None
```

### Function Structure
```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    # Implementation with clear variable names
    result = []
    return result
```

### Error Handling
```python
try:
    # Operation that may fail
    result = risky_operation()
    return result
except Exception as e:
    # Return user-friendly error messages
    return f"Operation failed: {str(e)}"
    # Or log and re-raise for critical errors
    # logger.error(f"Critical error: {str(e)}")
    # raise
```

### Flask Routes
```python
@app.route('/endpoint', methods=['POST'])
def handle_request():
    """Handle API request with proper error handling."""
    try:
        data = request.get_json()
        if not data or 'required_field' not in data:
            return jsonify({
                "code": 400,
                "content": "Missing required_field"
            }), 400
        # Process request
        return jsonify({"code": 0, "content": result})
    except Exception as e:
        return jsonify({"code": 500, "content": f"Error: {str(e)}"}), 500
```

### Database Operations
```python
from sqlalchemy import create_engine

# Use context manager for connections when possible
engine = create_engine(DB_URL, connect_args={'connect_timeout': 10})
df = pd.read_sql(query, engine)

# Handle empty results
if df.empty:
    return "No data found"
```

### Visualization (Matplotlib)
```python
import matplotlib.pyplot as plt

# Set Chinese font globally
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Save to static directory
save_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(image_path)
plt.close()  # Always close figures
```

### Tool Definition (Qwen-Agent)
```python
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('tool_name')
class ToolName(BaseTool):
    """Tool description for AI agent."""
    description = 'Clear description of what the tool does'
    parameters = [
        {
            'name': 'param_name',
            'type': 'string',
            'description': 'Parameter description',
            'required': True
        }
    ]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        # Implementation
        return "Result string"
```

### File Organization
- **app.py**: Flask routes and web endpoints
- **customer_operation_assistant.py**: Core AI agent and tools
- **chart_generator.py**: Visualization utilities
- **static/**: Generated charts and static assets
- **templates/**: HTML templates
- **data files**: CSV files in root directory

### Commit Messages (if needed)
- Use Chinese or English consistently
- Format: `type(scope): description`
- Examples: `feat: Add customer analysis endpoint`, `fix: Fix chart generation bug`

## Development Notes
- API keys and database credentials should use environment variables
- All generated files go to `static/` directory
- Charts use timestamps in filenames for cache busting
- AI tools must return markdown-formatted results
