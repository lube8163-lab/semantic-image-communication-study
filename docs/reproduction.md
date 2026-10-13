# 初回特徴pilotの再現

環境セットアップは [公開版手順](text-route-reproduction.md) を参照してください。

```powershell
./scripts/run-pilot.ps1 -Phase Encode -Run pilot-replay-001
./scripts/run-pilot.ps1 -Phase Decode -Run pilot-replay-001
./scripts/run-pilot.ps1 -Phase Evaluate -Run pilot-replay-001
```

EncodeとDecodeは別プロセスです。Decodeは元画像を読みません。Evaluateで元画像と比較します。量子化テストは `python -m pytest tests/test_packet.py` です。新しいrun名を使って公開結果を保持してください。元の取得・評価結果はresults/pilot-20261013、追加の1024px診断はresults/text-route-20261013/native-feature-checkにあります。
