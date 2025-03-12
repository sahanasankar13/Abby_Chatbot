# Gunicorn configuration file
import multiprocessing

# Bind to 0.0.0.0:5000
bind = "0.0.0.0:5000"

# Worker Options
# Use 2-4 workers per CPU core for web applications
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for longer AI operations
keepalive = 5

# Process Name
proc_name = "reproductive_health_chatbot"

# Logging
errorlog = "-"  # Log to stderr
loglevel = "info"
accesslog = "-"  # Log to stdout
access_log_format = '%({X-Forwarded-For}i)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Server Mechanics
preload_app = True  # Pre-load application code before worker processes fork

# Server Socket
backlog = 2048  # Maximum number of pending connections

# SSL Configuration
# For SSL in production, use a proper SSL termination at load balancer level
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
# ca_certs = "/path/to/ca_certs"

# Debugging
reload = False  # Set to False in production, True for development
spew = False  # Set to True to log all executed statements

# Server Hooks
def on_starting(server):
    server.log.info("Starting Reproductive Health Chatbot")

def on_exit(server):
    server.log.info("Shutting down Reproductive Health Chatbot")