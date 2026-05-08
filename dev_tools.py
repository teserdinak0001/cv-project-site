import sys
from datetime import datetime

class Logging():

    def __init__(self) -> None:
        """Default constructor for the Logging class. Logging is used for
        development and debugging purposes. It provides a simple way to print
        and view progress.
        Args:
            None
        Returns:
            None
        """
        self.color_codes = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
        }
        self.style_codes = {
        "bold": "1", "italic": "3", "underline": "4", "strikethrough": "9"
        }
        self.bg_color_codes = {
        "black": "40", "red": "41", "green": "42", "yellow": "43",
        "blue": "44", "magenta": "45", "cyan": "46", "white": "47"
        }

    def color_text(self, text, color=None, bg_color=None, style=None) -> str:
        """Formats text with ANSI escape codes for color, background color, and style using string inputs.
        Args:
            text (str): The text to format.
            color (str): The color to apply to the text.
            bg_color (str): The background color to apply to the text.
            style (str): The style to apply to the text.
        Returns:
            str: The formatted text.
        """
        codes = []
        if style in self.style_codes:
            codes.append(self.style_codes[style])
        if color in self.color_codes:
            codes.append(self.color_codes[color])
        if bg_color in self.bg_color_codes:
            codes.append(self.bg_color_codes[bg_color])

        if not codes:  # No valid formatting applied
            return text

        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def pprint(self, output_text, color=None, style=None, bg_color=None, runID='<Insert Preamble>') -> None:
        """Prints text with optional color, background color, and style using the color_text() method.
        Args:
            output_text (str): The text to print.
            color (str): The color to apply to the text.
            bg_color (str): The background color to apply to the text.
            style (str): The style to apply to the text.
        Returns:
            None
        """
        current_time = datetime.now()
        time_string = str(current_time.strftime("%H:%M:%S"))
        total_text = f"[{runID} - {time_string}] " + str(output_text)
        print(self.color_text(total_text, color=color, bg_color=bg_color, style=style))

    def pprogress(self, task,count, records, end_parameter='') -> None:
        """Prints a progress bar for the given task and current count and record count.
        Args:
            task (str): The task being performed.
            count (int): The current count of the task.
            records (int): The total number of records to process.
            end_parameter (str, optional): Additional parameters to display at the end of the progress bar.
        Returns:
            None
        """
        # Progress Bar:
        prog_bar_comp = '|' * int((count / records) * 100)
        prog_bar_incomp = '_' * (100 - int((count / records) * 100))

        if int((count / records) * 100) < 25:
            sys.stdout.write(self.color_text("\r{2}: ({3}/{4}) {0}{1} {5}".format(prog_bar_comp,
                          prog_bar_incomp,task,count,records,end_parameter),color='red'))
        elif int((count / records) * 100) >= 25 and int((count / records) * 100) < 75:
            sys.stdout.write(self.color_text("\r{2}: ({3}/{4}) {0}{1} {5}".format(prog_bar_comp,
                          prog_bar_incomp, task, count, records,end_parameter), color='yellow'))
        else:
            sys.stdout.write(self.color_text("\r{2}: ({3}/{4}) {0}{1} {5}".format(prog_bar_comp,
                          prog_bar_incomp, task, count, records,end_parameter), color='green'))

        sys.stdout.flush()
        if count == records: print('')