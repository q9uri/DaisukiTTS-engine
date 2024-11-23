"""
/multi_synthesis API のテスト
"""

from test.e2e.single_api.utils import gen_mora

from fastapi.testclient import TestClient


def test_post_multi_synthesis_200(client: TestClient) -> None:
    queries = [
        {
            "accent_phrases": [
                {
                    "moras": [
                        gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                        gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                    ],
                    "accent": 1,
                    "pause_mora": None,
                    "is_interrogative": False,
                }
            ],
            "speedScale": 1.0,
            "pitchScale": 1.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": 44100,
            "outputStereo": False,
            "kana": "テスト",
        },
        {
            "accent_phrases": [
                {
                    "moras": [
                        gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                        gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                        gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                    ],
                    "accent": 1,
                    "pause_mora": None,
                    "is_interrogative": False,
                }
            ],
            "speedScale": 1.0,
            "pitchScale": 1.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.2,
            "postPhonemeLength": 0.1,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": 44100,
            "outputStereo": False,
            "kana": "テストト",
        },
    ]
    response = client.post(
        "/multi_synthesis", params={"speaker": 888753760}, json=queries
    )
    assert response.status_code == 200

    # FileResponse 内の zip ファイルに圧縮された .wav から抽出された音声波形が一致する
    # FIXME: スナップショットテストを足す
    # NOTE: ZIP ファイル内の .wav に Linux-Windows 数値精度問題があるため解凍が必要
    assert response.headers["content-type"] == "application/zip"
    # from test.utility import summarize_wav_bytes
    # from syrupy.assertion import SnapshotAssertion
    # # zip 解凍
    # wav_summarys = map(lambda wav_byte: summarize_wav_bytes(wav_byte), wav_bytes)
    # wavs_summary = concatenate_func(wav_summarys)
    # assert snapshot == wavs_summary
