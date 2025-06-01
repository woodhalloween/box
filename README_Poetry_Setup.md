# 🦅 Poetry Local Setup

---

## ❖ ENGLISH VERSION: "Don't solve the setup. Win it."

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

If it doesn't work, you're not on the field.  
Also, add this to your shell:

- If you are using Zsh:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

- If you're using Bash:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

Test it:

```bash
poetry --version
```

---

### 2. Configure Project-Local Environment

> "We don't share benches. We fight alone, inside the box."

```bash
poetry config virtualenvs.in-project true
```

If you want to scope it to this repo only:

```bash
poetry config --local virtualenvs.in-project true
```

If needed, erase the existing poetry env:
```bash
poetry env remove python
```

---

### 3. Initialize the Field

```bash
poetry init
# or
poetry init --no-interaction
```

---

### 4. Lock the Formation

```bash
poetry lock
```

> "No second-guessing. Lock it down. Then run."

---

### 5. Install Your Winning Squad

```bash
poetry install
```

---

### 6. Add Dev Dependencies from requirements-dev.txt

```bash
poetry add --group dev $(< requirements-dev.txt)
```

Re-lock after major changes:

```bash
poetry lock
poetry install
```

---

### 7. Confirm the `.venv`

```bash
poetry env info --path
```

→ should be inside your project root, like:

```
./.venv
```

> "I shoot from this turf only."

---

### 8. PyCharm Sync

Use the `.venv/bin/python` as your interpreter.  
No excuses. Set it.

---

## ❖ 日本語版：「設定の問題？それ、勝つことで消してるから。」

### 1. Poetry を叩き込む

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

これすら通らないなら、  
――ピッチに立つ資格なし。

シェルにこの一行を入れろ：

- Zsh の場合：
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

- Bash の場合：
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

確認：

```bash
poetry --version
```

→ 表示されない？それ、敗北。終わり。

---

### 2. 仮想環境は“この場”で完結させろ

> 「ベンチは共有しない。私たちは、“ボックスの中”で一人で戦う。」

```bash
poetry config virtualenvs.in-project true
```

このプロジェクトだけで完結させたいなら：

```bash
poetry config --local virtualenvs.in-project true
```

過去の依存？  
邪魔。消せ。

```bash
poetry env remove python
```

---

### 3. ピッチを定義しろ

```bash
poetry init
# もしくは
poetry init --no-interaction
```

> 「最初から“構成”を描け。プレイはその後。」

---

### 4. ロックせよ。迷うな。

```bash
poetry lock
```

> 「迷ってる暇はない。勝つ前提でロックしろ。」

---

### 5. スカッドを並べろ

```bash
poetry install
```

ここまでで初めて、「戦える陣形」になる。

---

### 6. requirements-dev.txt から仲間を引き入れろ

```bash
poetry add --group dev $(< requirements-dev.txt)
```

構成が変わったら、再ロック。

```bash
poetry lock
poetry install
```

---

### 7. 仮想環境の“居場所”を確認

```bash
poetry env info --path
```

出力がこうなってなければ負け：

```
./.venv
```

> 「私は、この場でしか撃たない。」

---

### 8. PyCharmの同期

`.venv/bin/python` をインタープリタに設定。  
やれ。以上。  
できないなら、スタメン落ち。