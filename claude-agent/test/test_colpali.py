from claude_agent.main import app
from typer.testing import CliRunner


runner = CliRunner()


def test_colpali_search():
    result = runner.invoke(app, ["pa", "Find me a page on multi-head attention"])
    assert result.exit_code == 0
    assert "Attention Is All You Need" in result.stdout
    assert "4" or "5" in result.stdout
