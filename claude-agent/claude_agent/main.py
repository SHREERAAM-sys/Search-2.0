import typer
from claude_agent.base import chat_with_claude
from rich import print

app = typer.Typer(rich_markup_mode="rich")
state = {"verbose": False}
_logo = r"""
                          _                            _
  ___  ___  __ _ _ __ ___| |__   __ _  __ _  ___ _ __ | |_
 / __|/ _ \/ _ | '__/ __| '_ \ / _ |/ _` |/ _ \ '_ \| __|
 \__ \  __/ (_| | | | (__| | | | (_| | (_| |  __/ | | | |_
 |___/\___|\__,_|_|  \___|_| |_|\__,_|\__, |\___|_| |_|\__|
                                      |___/
"""
print(_logo)


@app.command()
def pa(query: str):
    """
    PDF Assistant: Handle PDF-related queries

    Including:

    - Searching PDF files
    - Embedding PDF files
    - Deleting embedded PDF files
    - Get all embedded PDF files

    Args:
        query (str): The user's query specifying the action or information needed.

    Examples:
        Search for a page on a specific topic:
            ```bash
            searchagent pa "Find me a page on neural ranking"
            ```

        Embed a PDF file:
            ```bash
            searchagent pa "Embed the paper at this path:
            /Users/friedahuang/ColPali_Efficient_Document_Retrieval_with_Vision_Language_Models.pdf"
            ```

        Embed multiple PDF Files:
            ```bash
            searchagent pa "Embed the papers at these paths: ['path1', 'path2']"
            ```

        Get all embedded PDF files:
            ```bash
            searchagent pa "Get all of my files"
            ```
    """
    chat_with_claude(query, verbose=state["verbose"])



@app.command()
def cs(query: str):
    """
    Code Search and Embed: Handle code search and code embedding tasks.

    Args:
        query (str): The user's query specifying the action or information needed.

    Examples:
        Search for a code snippet on a specific topic:
            ```bash
            searchagent cs "Find me code on sorting algorithms"
            ```

        Embed a code file:
            ```bash
            searchagent cs "Embed the code file at this path: /path/to/code.py"
            ```

        Embed multiple code files:
            ```bash
            searchagent cs "Embed the code files at these paths: ['/path/to/code1.py', '/path/to/code2.py']"
            ```
    """ 
   
    # Use chat_with_claude for code search as well
    chat_with_claude(query, verbose=state["verbose"])
    
    
@app.callback()
def main(verbose: bool = False):
    """
    Awesome SearchAgent CLI app
    """

    if verbose:
        state["verbose"] = True
        typer.echo("Verbose mode is enabled.")


if __name__ == "__main__":
    app()
