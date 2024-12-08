

# Get the service account token or return None
def get_token():
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: {e}")
        return None