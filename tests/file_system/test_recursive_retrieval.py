import pytest
import requests
from searchagent.file_system.base import FileSystemManager


@pytest.fixture
def pg_essays_html():
    base_url = "https://paulgraham.com"

    greatwork_html = "greatwork.html"
    hwh_html = "worked.html"
    worked_html = "worked.html"
    getideas_html = "getideas.html"

    greatwork_url = f"{base_url}/{greatwork_html}"
    hwh_url = f"{base_url}/{hwh_html}"
    worked_url = f"{base_url}/{worked_html}"
    getideas_url = f"{base_url}/{getideas_html}"

    greatwork = requests.get(greatwork_url).text
    hwh = requests.get(hwh_url).text
    worked = requests.get(worked_url).text
    getideas = requests.get(getideas_url).text

    return [greatwork, hwh, worked, getideas]


@pytest.fixture
def bitcoin_paper_pdf():
    url = "https://bitcoin.org/bitcoin.pdf"
    response = requests.get(url)
    return response.content


@pytest.fixture
def extracted_bitcoin_paper():
    return [
        "A purely peer-to-peer version of electronic cash would allow online payments to be sent "
        "directly from one party to another without going through a financial institution",
        "p = probability an honest node finds the next block",
        "1.Introduction",
        "2.Transactions",
        "3.Timestamp Server",
        "4.Proof-of-Work",
        "5.Network",
        "6.Incentive",
        "7.Reclaiming Disk Space",
        "8.Simplified Payment Verification",
        "9.Combining and Splitting Value",
        "10.Privacy",
        "11.Calculations",
    ]


@pytest.fixture
def format_extracted_text():
    def func(text):
        import re

        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = text.replace("\n", " ")
        return text

    return func


@pytest.fixture
def test_env(tmp_path, pg_essays_html, bitcoin_paper_pdf):
    from docx import Document

    """Files are of the type .html, .pdf, .docx, .csv, .md, .json
    | | d1
    |   | d1_subd1
    |       | d1_subd1_f1
    |       | d1_subd1_f2
    |       | d1_subd1_f3
    |       | d1_subd1_f4
    |   | d1_subd2
    |       | d1_subd2_f1
    |   | d1_f1
    |
    | | d2
    |   | d2_subd1
    |       | d2_subd1_f1
    |       | d2_subd1_f2
    |   | d2_f1
    """
    content = "content"

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"

    d1_subd1 = d1 / "subd1"
    d1_subd2 = d1 / "subd2"

    d2_subd1 = d2 / "subd1"

    d1.mkdir()
    d2.mkdir()
    d1_subd1.mkdir()
    d1_subd2.mkdir()
    d2_subd1.mkdir()

    d1_subd1_f1 = d1_subd1 / "greatwork.html"
    d1_subd1_f2 = d1_subd1 / "hwh.html"
    d1_subd1_f3 = d1_subd1 / "worked.html"
    d1_subd1_f4 = d1_subd1 / "getideas.html"

    d1_subd2_f1 = d1_subd2 / "bitcoin_paper.pdf"
    d1_f1 = d1 / "d1_f1.docx"

    d2_subd1_f1 = d2_subd1 / "d2_subd1_f1.csv"
    d2_subd1_f2 = d2_subd1 / "d2_subd1_f2.md"
    d2_f1 = d2 / "d2_f1.json"

    greatwork, hwh, worked, getideas = pg_essays_html
    bpaper = bitcoin_paper_pdf

    # Use python-docx to create documents
    document = Document()

    d1_subd1_f1.write_text(greatwork)
    d1_subd1_f2.write_text(hwh)
    d1_subd1_f3.write_text(worked)
    d1_subd1_f4.write_text(getideas)

    d1_subd2_f1.write_bytes(bpaper)
    document.add_heading(content, 0)
    document.save(d1_f1)

    d2_subd1_f1.write_text(content)
    d2_subd1_f2.write_text(content)
    d2_f1.write_text(content)

    return {
        "tmp_path": str(tmp_path),
        "files": {
            "d1_subd1_f1": d1_subd1_f1,
            "d1_subd1_f2": d1_subd1_f2,
            "d1_subd1_f3": d1_subd1_f3,
            "d1_subd1_f4": d1_subd1_f4,
            "d1_subd2_f1": d1_subd2_f1,
            "d1_f1": d1_f1,
            "d2_subd1_f1": d2_subd1_f1,
            "d2_subd1_f2": d2_subd1_f2,
            "d2_f1": d2_f1,
        },
        "content": content,
    }


@pytest.mark.xfail(raises=ValueError)
def test_invalid_dir():
    invalid_path = "/i/m/invalid"
    FileSystemManager(invalid_path)


@pytest.mark.parametrize("dir, expected_result", [("", 0), ("d1/subd2", 1)])
def test_list_files(test_env, dir, expected_result):
    tmp_path = test_env["tmp_path"]
    dirpath = f"{tmp_path}/{dir}"
    fm = FileSystemManager(dirpath)
    files = fm.list_files(["application/pdf"])
    assert len(files) == expected_result


def test_file_count(test_env):
    tmp_path = test_env["tmp_path"]
    files = test_env["files"]
    fm = FileSystemManager(tmp_path)
    retrieved_files = fm.scan()
    assert len(retrieved_files) == len(files)


def test_exclude_patterns_validity(test_env):
    tmp_path = test_env["tmp_path"]
    assert FileSystemManager(tmp_path, None).exclude_patterns == []


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "exclude_patterns", [["invalid"], [".docx", ".html", "", ".json"]]
)
def test_exclude_patterns_partial_invalid(test_env, exclude_patterns):
    tmp_path = test_env["tmp_path"]
    FileSystemManager(tmp_path, exclude_patterns)


@pytest.mark.parametrize(
    "exclude_patterns, expected_result",
    [
        (
            [".html"],
            [
                "bitcoin_paper.pdf",
                "d1_f1.docx",
                "d2_subd1_f1.csv",
                "d2_subd1_f2.md",
                "d2_f1.json",
            ],
        ),
        (
            [".docx", ".csv", ".json", ".md", ".pdf"],
            ["greatwork.html", "hwh.html", "worked.html", "getideas.html"],
        ),
        (
            [".docx", ".html", ".json"],
            ["bitcoin_paper.pdf", "d2_subd1_f1.csv", "d2_subd1_f2.md"],
        ),
    ],
)
def test_basic_exclude_patterns(test_env, exclude_patterns, expected_result):
    tmp_path = test_env["tmp_path"]
    fm = FileSystemManager(tmp_path, exclude_patterns)
    files = fm.scan()
    filenames = [fileInfo.metadata.filename for fileInfo in files]
    assert len(filenames) == len(expected_result)


def test_extract_text_from_pdf(
    test_env, extracted_bitcoin_paper, format_extracted_text
):
    NUM_PAGES = 9
    tmp_path = test_env["tmp_path"]
    filepath = f"{tmp_path}/d1/subd2/bitcoin_paper.pdf"
    fm = FileSystemManager(tmp_path)
    pages = fm.extract_text_from_pdf(filepath)
    contents = " ".join([format_extracted_text(p.content) for p in pages])

    assert len(pages) == NUM_PAGES
    for snippet in extracted_bitcoin_paper:
        assert snippet in contents


def test_extract_text_from_docx(test_env):
    tmp_path = test_env["tmp_path"]
    filepath = f"{tmp_path}/d1/d1_f1.docx"
    fm = FileSystemManager(tmp_path)
    pages = fm.extract_text_from_docx(filepath)

    number = pages[0].number
    content = pages[0].content.decode("utf-8")

    assert number == 1
    assert content == "content"
