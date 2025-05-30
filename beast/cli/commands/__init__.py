"""Command modules for the beast CLI."""

from beast.cli.commands import extract, predict, train

# dictionary of all available commands
COMMANDS = {
    'extract': extract,  # frame extraction
    'train': train,      # model training
    'predict': predict,  # model inference on images and videos
}
