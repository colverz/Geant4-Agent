# Geant4-Agent 浣跨敤鏂规硶涓庢湰鍦伴儴缃茶鏄?
## 1. 閫傜敤鑼冨洿

杩欎唤璇存槑闈㈠悜涓ょ被鍦烘櫙锛?
1. 鍦ㄦ湰鏈哄惎鍔?Web UI锛屽仛澶氳疆 first-pass 閰嶇疆鐢熸垚
2. 鍦ㄥ彟涓€鍙?Windows 鏈哄櫒涓婂鐜板綋鍓嶉」鐩?
褰撳墠鐗堟湰閫傚悎锛?
- 鏈湴楠岃瘉
- 鍐呴儴娴嬭瘯
- 鍙楁帶鐪熶汉璇曠敤

褰撳墠鐗堟湰涓嶉€傚悎锛?
- 鐩存帴浣滀负瀹屾暣 Geant4 宸ョ▼鐢熸垚鍣ㄥ彂甯?- 涓嶅彈鎺у叕寮€娴嬭瘯

---

## 2. 鎺ㄨ崘鐜

### 2.1 鎿嶄綔绯荤粺

鎺ㄨ崘锛?
- Windows 10 / 11

褰撳墠椤圭洰鐨勬棦鏈変娇鐢ㄦ柟寮忎笌鑴氭湰璺緞鏄庢樉鏄?Windows-first銆?
### 2.2 Python

鎺ㄨ崘锛?
- Python `3.10+`

褰撳墠鏈湴鐜瀹為檯杩愯鍦?Python 3.12銆?
### 2.3 纭欢

- 鍙仛鎺ㄧ悊涓?Web UI锛欳PU 鍙繍琛?- 鍋氭湰鍦拌缁冩垨楂橀鎺ㄧ悊锛氬缓璁?NVIDIA GPU

---

## 3. 鑾峰彇浠ｇ爜

鍋囪椤圭洰鐩綍涓猴細

```powershell
F:\geant4agent
```

鍚庣画鍛戒护榛樿閮藉湪浠撳簱鏍圭洰褰曟墽琛屻€?
---

## 4. 鍒涘缓铏氭嫙鐜

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

濡傛灉 PowerShell 闄愬埗鑴氭湰鎵ц锛屽彲鍏堣繍琛岋細

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

---

## 5. 瀹夎渚濊禆

褰撳墠浠撳簱娌℃湁缁熶竴閿佸畾鐨?`requirements.txt`锛屽缓璁畨瑁呮渶灏忚繍琛屼緷璧栥€?
### 5.1 CPU 瀹夎

```powershell
python -m pip install torch transformers safetensors
```

### 5.2 CUDA GPU 瀹夎锛堢ず渚嬶細CUDA 12.1锛?
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers safetensors
```

濡傛灉浣犵殑 CUDA 鐗堟湰涓嶅悓锛屽簲鏀规垚鍖归厤鐨?PyTorch wheel 婧愩€?
### 5.3 褰撳墠鏍稿績渚濊禆

杩愯鏃朵富瑕佷緷璧栵細

- `torch`
- `transformers`
- `safetensors`

Web UI 鏈韩鍩轰簬 Python 鏍囧噯搴?`http.server`锛屼笉渚濊禆 Flask/FastAPI銆?
---

## 6. 鍑嗗 LLM 閰嶇疆

閰嶇疆鐩綍锛?
- `nlu/llm_support/configs/`

褰撳墠鍙敤绀轰緥锛?
- `ollama_config.json`
- `ollama_config_fast.json`
- `ollama_config_expert_fast.json`
- `openai_compatible_config.example.json`
- `deepseek_api_config.example.json`

### 6.1 浣跨敤 Ollama

缂栬緫锛?
- `nlu/llm_support/configs/ollama_config.json`

纭繚浠ヤ笅瀛楁姝ｇ‘锛?
- `provider`
- `base_url`
- `model`

### 6.2 浣跨敤 OpenAI 鍏煎鎺ュ彛 / DeepSeek

鍙柊寤轰竴涓湰鍦扮鏈夐厤缃紝渚嬪锛?
- `nlu/llm_support/configs/deepseek_api.local.json`

绀轰緥锛?
```json
{
  "provider": "deepseek",
  "base_url": "https://api.deepseek.com",
  "chat_path": "/chat/completions",
  "model": "deepseek-chat",
  "timeout_s": 120,
  "api_key": "<YOUR_API_KEY>",
  "headers": {
    "Content-Type": "application/json"
  }
}
```

涓嶈鎶婄閽ラ厤缃彁浜ゅ埌鐗堟湰搴撱€?
---

## 7. 鍑嗗鏈湴妯″瀷

杩愯鏃朵細鑷姩妫€鏌ワ細

- structure 妯″瀷鐩綍
- `ner` 妯″瀷鐩綍

榛樿妫€鏌ラ€昏緫鍦細

- `nlu/runtime_components/model_preflight.py`

### 7.1 榛樿 structure 妯″瀷鍊欓€夌洰褰?
绯荤粺浼氫紭鍏堝鎵句互涓嬬洰褰曚箣涓€锛?
- `nlu/training/bert_lab/models/structure_controlled_v4c_e1`
- `nlu/training/bert_lab/models/structure_controlled_v3_e1`
- `nlu/training/bert_lab/models/structure_controlled_smoke`
- `nlu/training/bert_lab/models/structure_opt_v3`
- `nlu/training/bert_lab/models/structure_opt_v2`
- `nlu/training/bert_lab/models/structure`

### 7.2 榛樿 NER 妯″瀷鐩綍

- `nlu/training/bert_lab/models/ner`

### 7.3 妯″瀷鐩綍鏈€浣庤姹?
妯″瀷鐩綍鑷冲皯搴斿寘鍚細

- `config.json`
- tokenizer 璧勪骇涔嬩竴锛?  - `tokenizer.json`
  - `vocab.txt`
  - `vocab.json + merges.txt`
  - sentencepiece 鏂囦欢

### 7.4 閲嶈璇存槑

浠撳簱榛樿涓嶉檮甯﹁缁冨ソ鐨勬ā鍨嬫潈閲嶃€?
濡傛灉浣犲湪鍙︿竴鍙版満鍣ㄩ儴缃诧紝闇€瑕侊細

- 鎶婃湰鍦拌缁冨ソ鐨勬ā鍨嬬洰褰曞鍒惰繃鍘?- 鎴栧湪鏂版満鍣ㄩ噸鏂拌缁?
---

## 8. 鍚姩 Web UI

```powershell
python ui\web\server.py
```

鍚姩鍚庤闂細

- `http://127.0.0.1:8088`

鍚姩鏃ュ織浼氭墦鍗帮細

- 褰撳墠 provider
- 褰撳墠 model
- 褰撳墠 base URL
- model preflight 缁撴灉

濡傛灉 preflight 鏄剧ず锛?
- `structure_ok=false`
- 鎴?`ner_ok=false`

璇存槑妯″瀷鐩綍灏氭湭鍑嗗濂姐€?
---

## 9. 鏈湴浣跨敤娴佺▼

鎺ㄨ崘浣跨敤娴佺▼锛?
1. 鍚姩 Web UI
2. 纭褰撳墠 LLM config 鎸囧悜姝ｇ‘ provider
3. 杈撳叆鑷劧璇█闇€姹?4. 璺熼殢绯荤粺杩涜澶氳疆琛ュ弬
5. 鏌ョ湅杈撳嚭 JSON 涓庣敤鎴峰眰鎽樿

褰撳墠绯荤粺宸茬粡鏀寔锛?
- fuzzy 棣栬疆淇濆畧澶勭悊
- overwrite 纭 / 鎷掔粷
- graph family 鐨?family-aware 缂哄弬杩介棶

---

## 10. 浠呰繍琛?geometry 鐞嗚閮ㄥ垎

濡傛灉浣犲彧鎯宠繍琛?geometry DSL 涓庣悊璁洪獙璇侊紝涓嶉渶瑕佸惎鍔?UI銆?
```powershell
python -m builder.geometry.cli run_all --outdir builder/geometry/out --n_samples 200 --n_param_sets 100 --seed 7 --dataset builder/geometry/examples/coverage.csv
```

棰勬湡杈撳嚭锛?
- `builder/geometry/out/coverage_summary.json`
- `builder/geometry/out/coverage_checked.jsonl`
- `builder/geometry/out/feasibility_summary.json`
- `builder/geometry/out/ambiguity_summary.json`

杩欓儴鍒嗕笉渚濊禆 Geant4 杩愯鏃讹紝鍙仛鐞嗚灞傛鏌ャ€?
---

## 11. 杩愯鍥炲綊娴嬭瘯

### 11.1 褰撳墠宸查獙璇佺殑 live 闆嗗悎

```powershell
python scripts/run_casebank_regression.py --dataset docs/eval_casebank_multiturn_live_v2_12.json --config nlu/llm_support/configs/deepseek_api.local.json --outdir docs --baseline docs/casebank_baseline.json --min_confidence 0.6
```

鏈€杩戜竴娆＄粨鏋滐細

- `pass_rate = 1.0000`
- `internal_leak_turn_rate = 0.0000`

杈撳嚭鍐呭鍖呮嫭锛?
- JSON 鎶ュ憡
- 涓嫳鏂?PDF 鎶ュ憡
- LaTeX 婧愭枃浠?
### 11.2 妫€鏌?casebank 鍒嗗竷

```powershell
python scripts/verify_eval_casebank.py --dataset docs/eval_casebank_multiturn_live_v3_24.json
```

杩欎竴姝ュ彧妫€鏌ユ暟鎹泦鍒嗗竷锛屼笉璋冪敤 LLM銆?
---

## 12. 鏈湴璁粌 BERT

### 12.1 鏋勫缓 structure v2 鏁版嵁闆?
结构数据构建脚本已迁入本地归档工具目录 `legacy/tooling/`，不再作为仓库主链路的一部分跟踪。

### 12.2 璁粌 structure 妯″瀷

训练配置保留在：

- `nlu/training/bert_lab/configs/structure_train_v2.json`

璁粌杈撳嚭鏀惧埌锛?
- `nlu/training/bert_lab/models/structure_controlled_v5_e2`

涔嬪悗 runtime preflight 浼氳嚜鍔ㄥ彂鐜板畠銆?
---

## 13. 鍦ㄥ彟涓€鍙?Windows 鏈哄櫒閮ㄧ讲

鎺ㄨ崘椤哄簭锛?
1. 澶嶅埗浠ｇ爜浠撳簱
2. 鍒涘缓 `.venv`
3. 瀹夎 `torch + transformers + safetensors`
4. 澶嶅埗鏈湴璁粌濂界殑妯″瀷鍒?`nlu/training/bert_lab/models/`
5. 鍑嗗 LLM 閰嶇疆鏂囦欢
6. 鍚姩 `python ui\\web\\server.py`
7. 鎵撳紑娴忚鍣ㄨ闂?`http://127.0.0.1:8088`

### 濡傛灉 LLM 鍦ㄨ繙绔?
渚嬪锛?
- 杩滅鏈哄櫒杩愯 Ollama
- 鏈満閫氳繃 SSH 杞彂鍒?`11434`

鍙渶瑕佹妸閰嶇疆鏂囦欢涓殑 `base_url` 鏀瑰埌鏈満杞彂鍦板潃锛屼緥濡傦細

- `http://127.0.0.1:11434`

椤圭洰鏈韩涓嶈姹傛ā鍨嬩笌 UI 璺戝湪鍚屼竴鍙版満鍣ㄣ€?
---

## 14. 甯歌闂

### 14.1 UI 鑳芥墦寮€锛屼絾瀵硅瘽涓嶅伐浣?
浼樺厛妫€鏌ワ細

1. LLM 閰嶇疆鏄惁姝ｇ‘
2. API key 鏄惁鏈夋晥
3. model preflight 鏄惁閫氳繃
4. structure 涓?NER 妯″瀷鐩綍鏄惁閮藉瓨鍦?
### 14.2 鍥炲寰堟満姊?
杩欓€氬父涓嶆槸閮ㄧ讲鏁呴殰锛岃€屾槸锛?
- 褰撳墠鍔ㄤ綔绾фā鏉垮湪璧蜂繚鎶や綔鐢?- 鎴?LLM 鑷劧鍖栬鎷掔粷鍚庡洖閫€鍒板畨鍏ㄦā鏉?
### 14.3 閰嶇疆鑳界悊瑙ｏ紝浣嗘煇浜涘瓧娈典笉鎻愪氦

褰撳墠涓婚摼宸茬粡淇宸茬煡 P0 鎻愪氦闂銆傝嫢浠嶅嚭鐜帮紝浼樺厛妫€鏌ワ細

- 鏄惁鏄?fuzzy 棣栬疆
- 鏄惁琚?lock 鎷︽埅
- 鏄惁琚?explicit target filter 涓㈠純
- 鏄惁琚?Gate 鎷掔粷

### 14.4 涓轰粈涔堝綋鍓嶄笉鐩存帴杈撳嚭瀹屾暣 Geant4 宸ョ▼

鍥犱负褰撳墠浼樺厛绾ф槸锛?
- 鍏堝仛瀵硅瘽闂幆
- 鍏堝仛閰嶇疆姝ｇ‘鎬?- 鍏堜繚璇佸彲瑙ｉ噴鍜屽彲杩介棶

瀹屾暣宸ョ▼鐢熸垚浠嶅湪鍚庣画闃舵銆?
---

## 15. 閮ㄧ讲寤鸿

濡傛灉浣犵殑鐩爣鏄細

- 鏈湴楠岃瘉
- 鍐呴儴娴嬭瘯
- 鍙楁帶璇曠敤

褰撳墠鐗堟湰宸茬粡鍙互浣跨敤銆?
濡傛灉浣犵殑鐩爣鏄細

- 闈㈠悜澶栭儴鐢ㄦ埛鐩存帴寮€鏀?- 瀵瑰鏉傚嚑浣曞仛楂樿鐩栨壙璇?- 鐩存帴杈撳嚭瀹屾暣 Geant4 宸ョ▼

褰撳墠鐗堟湰杩樹笉澶燂紝闇€瑕佺户缁墿澶?live casebank 骞惰ˉ澶嶆潅鍑犱綍鍥炲綊銆?
