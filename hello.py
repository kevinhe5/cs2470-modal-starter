"""First Modal example: modal run hello.py"""
import modal

app = modal.App("cs2470-hello")


@app.function()
def square(x: int) -> int:
    print("Running on a remote CPU worker!")
    return x**2


@app.local_entrypoint()
def main():
    # main runs on your laptop; .remote() sends the work to Modal.
    print("The square is", square.remote(42))
