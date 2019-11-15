## How to work with things on Nyla conveniently

### Running Jupyter notebooks

#### On Nyla:
1. Start a tmux (or screen) session
```bash
tmux new -s notebook_server
```

2. Activate your virtual environment if needed
```bash
source /path/to/your/venv/bin/activate
```

3. Start the notebook server (8915 used as random example)
```bash
jupyter notebook --no-browser --port 8915
```

#### On your local machine:

1. Listen to notebook updates on specified port
```bash
ssh -fNL 8915:localhost:8915 user@nyla.stanford.edu
```

2. Copy the link that was produced in step 3 (with the access token) and paste into browser window

#### When you're done

1. Exit the virtual environment
```bash
deactivate
```
