import sys

def progress_bar(current, total, bar_length=40):
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = "█" * filled + "-" * (bar_length - filled)
    percent = fraction * 100
    sys.stdout.write(f"\r|{bar}| {percent:6.2f}%\n")
    sys.stdout.flush()

    if current == total:
        print() 
