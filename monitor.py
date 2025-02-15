import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

# Install rich traceback handler
install(show_locals=True)

# Configure rich console
console = Console()

# Configure logging with rich handler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

logger = logging.getLogger("ATS-Monitor")

class LogHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_position = 0
        
    def on_modified(self, event):
        if event.src_path.endswith('.log'):
            with open(event.src_path, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                
                for line in new_lines:
                    if 'ERROR' in line:
                        console.print(f"[red]{line.strip()}[/red]")
                    elif 'WARNING' in line:
                        console.print(f"[yellow]{line.strip()}[/yellow]")
                    elif 'DEBUG' in line:
                        console.print(f"[blue]{line.strip()}[/blue]")
                    else:
                        console.print(line.strip())

def start_monitoring():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up file handler for logging
    file_handler = logging.FileHandler("logs/ats_optimizer.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Start file monitoring
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, str(log_dir), recursive=False)
    observer.start()
    
    logger.info("Started monitoring logs...")
    console.print("[green]Log monitoring started. Watching for application events...[/green]")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped monitoring logs.")
    observer.join()

if __name__ == "__main__":
    start_monitoring()
