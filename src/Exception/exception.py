import sys

def error_message_details(error, error_details: sys = None):
    """
    Construct detailed error message with filename and line number.
    If error_details is None or traceback is not available, fallback values are used.
    """
    try:
        if error_details:
            exc_type, exc_obj, exc_tb = error_details.exc_info()
            if exc_tb:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_no = exc_tb.tb_lineno
            else:
                file_name = "<unknown>"
                line_no = 0
        else:
            file_name = "<unknown>"
            line_no = 0

        error_message = f"Error occurred in python script [{file_name}] at line [{line_no}]: {str(error)}"
        return error_message

    except Exception as e:
        # fallback message if traceback fails
        return f"Error occurred but traceback info unavailable: {str(error)}"


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys = None):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details=error_details)

    def __str__(self):
        return self.error_message
