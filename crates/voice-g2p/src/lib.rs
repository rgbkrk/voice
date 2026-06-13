pub mod espeak;
pub mod lexicon;
pub mod number;
pub mod stress;
pub mod tagger;
pub mod token;
pub mod tokenizer;

use std::collections::HashMap;
use std::sync::OnceLock;

use espeak::EspeakFallback;
use lexicon::Lexicon;
use stress::{apply_stress, CONSONANTS, NON_QUOTE_PUNCTS, PRIMARY_STRESS, SUBTOKEN_JUNKS, VOWELS};
use token::{merge_tokens, MToken, TokenContext};
use tokenizer::TokenOrGroup;

#[derive(Debug, thiserror::Error)]
pub enum G2pError {
    #[error("legacy espeak-ng helper not found")]
    EspeakNotFound,
    #[error("legacy espeak-ng helper failed: {0}")]
    EspeakFailed(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration for external tool paths used by the G2P pipeline.
#[derive(Debug, Clone)]
pub struct G2PConfig {
    /// Legacy path to the `espeak-ng` binary.
    ///
    /// Runtime OOV fallback is embedded and does not require this binary; the
    /// field is retained so existing callers can keep constructing
    /// `G2PConfig` without source changes.
    pub espeak_path: String,
}

impl Default for G2PConfig {
    fn default() -> Self {
        Self {
            espeak_path: "espeak-ng".to_string(),
        }
    }
}

/// The main G2P pipeline, ported from misaki's `en.G2P.__call__()`.
pub struct G2P {
    lexicon: Lexicon,
    fallback: EspeakFallback,
    unk: String,
    overrides: HashMap<String, String>,
}

fn global_g2p() -> &'static G2P {
    static INSTANCE: OnceLock<G2P> = OnceLock::new();
    INSTANCE.get_or_init(G2P::new)
}

impl G2P {
    pub fn new() -> Self {
        Self::with_config(G2PConfig::default())
    }

    pub fn with_config(config: G2PConfig) -> Self {
        Self {
            lexicon: Lexicon::new(),
            fallback: EspeakFallback::with_path(config.espeak_path),
            unk: String::new(),
            overrides: Self::builtin_overrides(),
        }
    }

    /// Words whose default lexicon/espeak phonemes are wrong or misleading.
    fn builtin_overrides() -> HashMap<String, String> {
        const ENTRIES: &[(&str, &str)] = &[
            ("acl", "ˈA sˈi ˈɛl"),
            ("api", "ˈA pˈi ˌI"),
            ("apis", "ˈA pˈi ˈIz"),
            ("automerge", "ˈɔTO mˈɜɹʤ"),
            ("automunge", "ˈɔTO mˈʌnʤ"),
            ("aws", "ˈA dˈʌbᵊlju ˈɛs"),
            ("anywidget", "ˈɛni wˌɪʤət"),
            ("bilstm", "bˈI ˈɛl ˈɛs tˈi ˈɛm"),
            ("byoc", "bˈi wˈI ˈO sˈi"),
            ("chatgpt", "ʧˈæt ʤˈi pˈi tˈi"),
            ("cli", "sˈi ˈɛl ˌI"),
            ("clis", "sˈi ˈɛl ˈIz"),
            ("cloudflare", "klˈWd flˈɛɹ"),
            ("coreaudio", "kˈɔɹ ˈɔdiO"),
            ("coreml", "kˈɔɹ ˈɛm ˈɛl"),
            ("cpal", "sˈi pˈi ˈA ˈɛl"),
            ("crdt", "sˈi ˈɑɹ dˈi tˈi"),
            ("crdts", "sˈi ˈɑɹ dˈi tˈiz"),
            ("csr", "sˈi ˈɛs ˈɑɹ"),
            ("css", "sˈi ˈɛs ˈɛs"),
            ("cuda", "kˈudə"),
            ("d1", "dˈi wˈʌn"),
            ("deno", "dˈinO"),
            ("demo", "dˈɛmO"),
            ("demos", "dˈɛmOz"),
            ("demultiplex", "dˌimˈʌltɪplɛks"),
            ("demultiplexing", "dˌimˈʌltɪplɛksɪŋ"),
            ("demux", "dˌimˈʌks"),
            ("demuxing", "dˌimˈʌksɪŋ"),
            ("dft", "dˈi ˈɛf tˈi"),
            ("dtls", "dˈi tˈi ˈɛl ˈɛs"),
            ("duckdb", "dˈʌk dˈi bˈi"),
            ("esbuild", "ˈi ˈɛs bˌɪld"),
            ("eslint", "ˈi ˈɛs lˌɪnt"),
            ("espeak", "ˈi spˌik"),
            ("fastapi", "fˈæst ˈA pˈi ˌI"),
            ("flac", "flˈæk"),
            ("gguf", "ʤˈi ʤˈi jˈu ˈɛf"),
            ("grpc", "ʤˈi ˈɑɹ pˈi sˈi"),
            ("gpt", "ʤˈi pˈi tˈi"),
            ("http", "ˈAʧ tˈi tˈi pˈi"),
            ("https", "ˈAʧ tˈi tˈi pˈi ˈɛs"),
            ("html", "ˈAʧ tˈi ˈɛm ˈɛl"),
            ("htmliframeelement", "ˈAʧ tˈi ˈɛm ˈɛl ˌI fɹˌAm ˈɛləmənt"),
            ("id", "ˈI dˈi"),
            ("ids", "ˈI dˈiz"),
            ("idb", "ˌI dˈi bˈi"),
            ("iframe", "ˌI fɹˌAm"),
            ("ios", "ˈI ˈO ˈɛs"),
            ("indexeddb", "ˈɪndɛkst dˈi bˈi"),
            ("ipc", "ˌI pˈi sˈi"),
            ("ipykernel", "ˈI pˈI kˌɜɹnᵊl"),
            ("ipython", "ˌI pˈIθˌɑn"),
            ("ipywidgets", "ˌI pˈI wˌɪʤəts"),
            ("isort", "ˈI sˌɔɹt"),
            ("istft", "ˈI ˈɛs tˈi ˈɛf tˈi"),
            ("jax", "ʤˈæks"),
            ("json", "ʤˌA sˈæhn"),
            ("jsonrpc", "ʤˌA sˈæhn ˈɑɹ pˈi sˈi"),
            ("jsx", "ʤˈA ˈɛs ˈɛks"),
            ("jupyter", "ʤˈupɪTəɹ"),
            ("jwt", "ʤˈA dˈʌbᵊlju tˈi"),
            ("jwts", "ʤˈA dˈʌbᵊlju tˈiz"),
            ("katex", "kˈA tˌɛk"),
            ("kernelspec", "kˈɜɹnᵊl spˌɛk"),
            ("kokoro", "kˈOkəɹO"),
            ("kokoro-82m", "kˈOkəɹO ˈATi tˈu ˈɛm"),
            ("kubernetes", "kˌubəɹnˈɛtiz"),
            ("kubectl", "kjˈub kˈʌdᵊl"),
            ("latex", "lˈA tˌɛk"),
            ("lfs", "ˈɛl ˈɛf ˈɛs"),
            ("lstm", "ˈɛl ˈɛs tˈi ˈɛm"),
            ("macos", "mˈæk ˈO ˈɛs"),
            ("matplotlib", "mˈæt plˌɑt lˌɪb"),
            ("mathjax", "mˈæθ ʤˌæks"),
            ("mcp", "ˈɛm sˈi pˈi"),
            ("mcps", "ˈɛm sˈi pˈiz"),
            ("mdx", "ˈɛm dˈi ˈɛks"),
            ("micropip", "mˈIkɹO pˌɪp"),
            ("mimebundle", "mˈIm bˌʌndᵊl"),
            ("mlx", "ˈɛm ˈɛl ˈɛks"),
            ("mmap", "ˈɛm mˌæp"),
            ("msw", "ˈɛm ˈɛs dˈʌbᵊlju"),
            ("mypy", "mˈI pˌI"),
            ("nbconvert", "ˈɛn bˈi kˌɑnvɜɹt"),
            ("nbformat", "ˈɛn bˈi fˌɔɹmæt"),
            ("neuphonic", "nˈu fˌɑnɪk"),
            ("neutts", "nˈu tˈi tˈi ˈɛs"),
            ("next.js", "nˈɛkst ʤˈA ˈɛs"),
            ("nextjs", "nˈɛkst ʤˈA ˈɛs"),
            ("nginx", "ˈɛnʤən ˌɛks"),
            ("node.js", "nˈOd ʤˈA ˈɛs"),
            ("nodejs", "nˈOd ʤˈA ˈɛs"),
            ("nteract", "ˈɛntəɹˌækt"),
            ("nteract-dev", "ˈɛntəɹˌækt dˈɛv"),
            ("numpy", "nˈʌm pˌI"),
            ("oauth", "ˌO ˈɔθ"),
            ("ogg", "ˈɑɡ"),
            ("oidc", "ˈO ˌI dˈi sˈi"),
            ("onnx", "ˈɑnɪks"),
            ("openai-codex", "ˌOpᵊn ˈAˌI kˈOdˌɛks"),
            ("openapi", "ˈOpᵊn ˈA pˈi ˌI"),
            ("opfs", "ˈO pˈi ˈɛf ˈɛs"),
            ("outerbounds", "ˈWTəɹ bˈWndz"),
            ("outputidchanges", "ˈWtpˌʊt ˌI dˌi ʧˈAnʤᵻz"),
            ("pnpm", "pˈi ˈɛn pˈi ˈɛm"),
            ("postgres", "pˈOstɡɹɛs"),
            ("postgresql", "pˈOst ɡɹˈɛs kjˈu ˈɛl"),
            ("pcm", "pˈi sˈi ˈɛm"),
            ("protobuf", "pɹˈOTO bˌʌf"),
            ("pwa", "pˈi dˈʌbᵊlju ˈA"),
            ("put_blob", "pˌʊt blˈɑb"),
            ("putblob", "pˌʊt blˈɑb"),
            ("pyodide", "pˈI ə dˌId"),
            ("pyarrow", "pˈI ˈɛɹO"),
            ("pypi", "pˈI pˈi ˌI"),
            ("pyo3", "pˈI ˈO θɹˈi"),
            ("pytest", "pˈI tˌɛst"),
            ("qqbot", "kjˈu kjˈu bˌɑt"),
            ("r2", "ˈɑɹ tˈu"),
            ("js", "ʤˈA ˈɛs"),
            ("todo", "tˈudu"),
            // Developer acronyms and initialisms
            ("ipynb", "nˈOtbˌʊk fˈIl"),
            ("pr", "pˈi ˈɑɹ"),
            ("prs", "pˈi ˈɑɹz"),
            ("pytorch", "pˈI tˌɔɹʧ"),
            ("rpc", "ˈɑɹ pˈi sˈi"),
            ("rpcs", "ˈɑɹ pˈi sˈiz"),
            ("rodio", "ɹˈOdiˌO"),
            ("rtcp", "ˈɑɹ tˈi sˈi pˈi"),
            ("rtp", "ˈɑɹ tˈi pˈi"),
            ("rxjs", "ˈɑɹ ˈɛks ʤˈA ˈɛs"),
            ("repr-llm", "ɹˈɛpəɹ ˈɛl ˈɛl ˈɛm"),
            ("runt-nightly", "ɹˈʌnt nˈItli"),
            ("runtimed", "ɹˈʌntIm dˈi"),
            ("runtimed-wasm", "ɹˈʌntIm dˈi wˈæzəm"),
            ("runtime_peer", "ɹˈʌntIm pˈɪɹ"),
            ("runtime-peer", "ɹˈʌntIm pˈɪɹ"),
            ("s3", "ˈɛs θɹˈi"),
            ("safetensors", "sˈAf tˌɛnsəɹz"),
            ("scikit-learn", "sˈɪkɪt lˌɜɹn"),
            ("scikitlearn", "sˈɪkɪt lˌɜɹn"),
            ("scipy", "sˈI pˌI"),
            ("sdp", "ˈɛs dˈi pˈi"),
            ("serde", "sˈɜɹdˌi"),
            ("sklearn", "ˈɛs kˈA lˌɜɹn"),
            ("sqlite", "ˌɛs kjˈu ˌɛl lˈIt"),
            ("scss", "ˈɛs sˈi ˈɛs ˈɛs"),
            ("srtp", "ˈɛs ˈɑɹ tˈi pˈi"),
            ("ssr", "ˈɛs ˈɛs ˈɑɹ"),
            ("stft", "ˈɛs tˈi ˈɛf tˈi"),
            ("stt", "ˈɛs tˈi tˈi"),
            ("supabase", "sˈupə bˌAs"),
            ("swc", "ˈɛs dˈʌbᵊlju sˈi"),
            ("tailwindcss", "tˈAl wˈɪnd sˈi ˈɛs ˈɛs"),
            ("tauri", "tˈW ɹˌi"),
            ("tex", "tˈɛk"),
            ("tokenizer", "tˈOkᵊn ˌIzəɹ"),
            ("tokenizers", "tˈOkᵊn ˌIzəɹz"),
            ("toml", "tˈɑmᵊl"),
            ("tokio", "tˈOkiˌO"),
            ("tsconfig", "tˈi ˈɛs kˌɑnfˈɪɡ"),
            ("tsx", "tˈi ˈɛs ˈɛks"),
            ("tts", "tˈi tˈi ˈɛs"),
            ("typescript", "tˈIp skɹˌɪpt"),
            ("ui", "jˈu ˈI"),
            ("ulid", "jˈu lˌɪd"),
            ("url", "jˈu ˈɑɹ ˈɛl"),
            ("urls", "jˈu ˈɑɹ ˈɛlz"),
            ("uri", "jˈu ˈɑɹ ˌI"),
            ("uris", "jˈu ˈɑɹ ˈIz"),
            ("utf8view", "jˈu tˈi ˈɛf ˈAt vjˈu"),
            ("ux", "jˈu ˈɛks"),
            ("uuid", "jˈu jˈu ˌI dˈi"),
            ("uuids", "jˈu jˈu ˌI dˈiz"),
            ("vad", "vˈi ˈA dˈi"),
            ("vads", "vˈi ˈA dˈiz"),
            ("vite", "vˈit"),
            ("vitest", "vˈi tˌɛst"),
            ("vue", "vjˈu"),
            ("vscode", "vˈi ˈɛs kˈOd"),
            ("wav", "wˈAv"),
            ("wavs", "wˈAvz"),
            ("wasm", "wˈæzəm"),
            ("webrtc", "wˈɛb ˈɑɹ tˈi sˈi"),
            ("websocket", "wˈɛb sˌɑkət"),
            ("websockets", "wˈɛb sˌɑkəts"),
            ("wifi", "wˈI fˌI"),
            ("xai", "ˈɛks ˈA ˌI"),
            ("yaml", "jˈæmᵊl"),
            ("yjs", "wˈI ʤˈA ˈɛs"),
            ("zeromq", "zˈɪɹO ˈɛm kjˈu"),
            ("zmq", "zˈi ˈɛm kjˈu"),
        ];
        ENTRIES
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect()
    }

    /// Set custom word-to-phoneme overrides (builder pattern).
    ///
    /// Overrides map lowercase words to phoneme strings, checked before
    /// the lexicon and embedded fallback.
    pub fn with_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        self.overrides.extend(overrides);
        self
    }

    /// Full pipeline: text -> phoneme string.
    ///
    /// Mirrors misaki `G2P.__call__()` from en.py:679-738.
    pub fn convert(&self, text: &str) -> Result<String, G2pError> {
        // 1. Tokenize and POS-tag (embedded perceptron tagger)
        let tokens = tokenizer::tokenize(text);

        // 2. fold_left: merge non-head tokens
        let tokens = tokenizer::fold_left(tokens);

        // 3. retokenize: subtokenize, handle punctuation/currency
        let mut items = tokenizer::retokenize(tokens);

        // 4. Right-to-left resolution with TokenContext
        let mut ctx = TokenContext::default();

        for item in items.iter_mut().rev() {
            match item {
                TokenOrGroup::Single(ref mut w) => {
                    self.resolve_single_token(w, &ctx);
                    ctx = Self::token_context(&ctx, w.phonemes.as_deref(), w);
                }
                TokenOrGroup::Group(ref mut group) => {
                    self.resolve_group(group, &ctx);
                    if let Some(first) = group.first() {
                        ctx = Self::token_context(&ctx, first.phonemes.as_deref(), first);
                    }
                }
            }
        }

        // 5. Merge groups into single tokens
        let tokens: Vec<MToken> = items
            .into_iter()
            .map(|item| match item {
                TokenOrGroup::Single(tok) => tok,
                TokenOrGroup::Group(group) => merge_tokens(&group, Some(&self.unk)),
            })
            .collect();

        // 6. Legacy conversion: ɾ->T, ʔ->t
        let result: String = tokens
            .iter()
            .map(|tk| {
                let ps = match &tk.phonemes {
                    Some(p) => p.replace('ɾ', "T").replace('ʔ', "t"),
                    None => self.unk.clone(),
                };
                format!("{}{}", ps, tk.whitespace)
            })
            .collect();

        Ok(result)
    }

    /// Resolve a single (non-grouped) token.
    fn resolve_single_token(&self, w: &mut MToken, ctx: &TokenContext) {
        if w.phonemes.is_some() {
            return;
        }

        // Check custom overrides before lexicon/embedded fallback.
        let lookup_key = w.text.to_lowercase();
        if let Some(ps) = self.overrides.get(&lookup_key) {
            w.phonemes = Some(ps.clone());
            w.underscore.rating = Some(5); // highest priority
            return;
        }
        let (ps, rating) = self.lexicon.call(
            &w.text,
            w.underscore.alias.as_deref(),
            &w.tag,
            w.underscore.stress,
            w.underscore.currency,
            w.underscore.is_head,
            &w.underscore.num_flags,
            ctx,
        );
        if let Some(ps) = ps {
            w.phonemes = Some(ps);
            w.underscore.rating = rating;
            return;
        }

        if let Some((ps, rating)) = self.fallback.convert_word(&w.text) {
            w.phonemes = Some(ps);
            w.underscore.rating = Some(rating);
        }
    }

    /// Resolve a group of subtokens using the left-expand/right-shrink algorithm.
    ///
    /// Ported from en.py:694-731.
    fn resolve_group(&self, group: &mut [MToken], ctx: &TokenContext) {
        // Check overrides for the whole merged text before the expand/shrink loop
        let merged_text: String = group.iter().map(|tk| tk.text.as_str()).collect();
        let lookup_key = merged_text.to_lowercase();
        if let Some(ps) = self.overrides.get(&lookup_key) {
            group[0].phonemes = Some(ps.clone());
            group[0].underscore.rating = Some(5);
            for tk in group.iter_mut().skip(1) {
                tk.phonemes = Some(String::new());
                tk.underscore.rating = Some(5);
            }
            return;
        }

        let n = group.len();
        let mut left = 0;
        let mut right = n;
        let mut should_fallback = false;

        while left < right {
            let has_existing = group[left..right]
                .iter()
                .any(|tk| tk.underscore.alias.is_some() || tk.phonemes.is_some());

            let (ps, rating) = if has_existing {
                (None, None)
            } else {
                let merged = merge_tokens(&group[left..right], None);
                if let Some(ps) = self.overrides.get(&merged.text.to_lowercase()) {
                    (Some(ps.clone()), Some(5))
                } else {
                    self.lexicon.call(
                        &merged.text,
                        merged.underscore.alias.as_deref(),
                        &merged.tag,
                        merged.underscore.stress,
                        merged.underscore.currency,
                        merged.underscore.is_head,
                        &merged.underscore.num_flags,
                        ctx,
                    )
                }
            };

            if let Some(ps) = ps {
                group[left].phonemes = Some(ps);
                group[left].underscore.rating = rating;
                for x in &mut group[left + 1..right] {
                    x.phonemes = Some(String::new());
                    x.underscore.rating = rating;
                }
                right = left;
                left = 0;
            } else if left + 1 < right {
                left += 1;
            } else {
                right -= 1;
                let tk = &mut group[right];
                if tk.phonemes.is_none() {
                    if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                        tk.phonemes = Some(String::new());
                        tk.underscore.rating = Some(3);
                    } else {
                        should_fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }

        if should_fallback {
            for tk in group.iter_mut() {
                if tk.phonemes.is_some() {
                    continue;
                }
                if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                    tk.phonemes = Some(String::new());
                    tk.underscore.rating = Some(3);
                } else {
                    self.resolve_single_token(tk, ctx);
                }
            }
        }

        Self::resolve_tokens(group);
    }

    /// Update TokenContext based on resolved phonemes and token.
    ///
    /// Ported from en.py:646-650.
    fn token_context(ctx: &TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
        let mut vowel = ctx.future_vowel;

        if let Some(ps) = ps {
            for c in ps.chars() {
                let is_vowel = VOWELS.contains(c);
                let is_consonant = CONSONANTS.contains(c);
                let is_punct = NON_QUOTE_PUNCTS.contains(c);

                if is_vowel || is_consonant || is_punct {
                    vowel = if is_punct { None } else { Some(is_vowel) };
                    break;
                }
            }
        }

        let future_to = matches!(token.text.as_str(), "to" | "To")
            || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));

        TokenContext {
            future_vowel: vowel,
            future_to,
        }
    }

    /// Normalize stress across a group of resolved subtokens.
    ///
    /// Ported from en.py:652-677.
    fn resolve_tokens(tokens: &mut [MToken]) {
        if tokens.is_empty() {
            return;
        }

        let text: String = tokens
            .iter()
            .enumerate()
            .map(|(i, tk)| {
                if i < tokens.len() - 1 {
                    format!("{}{}", tk.text, tk.whitespace)
                } else {
                    tk.text.clone()
                }
            })
            .collect();

        let has_space = text.contains(' ') || text.contains('/');
        let char_classes: std::collections::HashSet<u8> = text
            .chars()
            .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
            .map(|c| {
                if c.is_alphabetic() {
                    0
                } else if c.is_ascii_digit() {
                    1
                } else {
                    2
                }
            })
            .collect();
        let prespace = has_space || char_classes.len() > 1;

        let n = tokens.len();
        for (i, tk) in tokens.iter_mut().enumerate() {
            if tk.phonemes.is_none() {
                let last = i == n - 1;
                if last
                    && tk.text.len() == 1
                    && NON_QUOTE_PUNCTS.contains(tk.text.chars().next().unwrap_or(' '))
                {
                    tk.phonemes = Some(tk.text.clone());
                    tk.underscore.rating = Some(3);
                } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                    tk.phonemes = Some(String::new());
                    tk.underscore.rating = Some(3);
                }
            } else if i > 0 && !tk.underscore.prespace {
                tk.underscore.prespace = prespace;
            }
        }

        if prespace {
            return;
        }

        let indices: Vec<(bool, usize, usize)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tk)| {
                tk.phonemes.as_ref().filter(|p| !p.is_empty()).map(|p| {
                    let has_primary = p.contains(PRIMARY_STRESS);
                    let weight = token::stress_weight(Some(p));
                    (has_primary, weight, i)
                })
            })
            .collect();

        if indices.len() == 2 && tokens[indices[0].2].text.len() == 1 {
            let i = indices[1].2;
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
            return;
        }

        if indices.len() < 2 {
            return;
        }
        let primary_count: usize = indices.iter().filter(|(b, _, _)| *b).count();
        if primary_count <= indices.len().div_ceil(2) {
            return;
        }

        let mut sorted = indices.clone();
        sorted.sort();
        let half = sorted.len() / 2;
        for &(_, _, i) in &sorted[..half] {
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
        }
    }
}

impl Default for G2P {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Public API (backward-compatible)
// ---------------------------------------------------------------------------

/// Convert English text to a Kokoro-compatible phoneme string.
///
/// Uses misaki-style dictionary lookup with embedded fallback for unknown words.
pub fn english_to_phonemes(text: &str) -> Result<String, G2pError> {
    global_g2p().convert(text)
}

/// Convert English text to phonemes with custom word overrides.
///
/// Overrides map lowercase words to phoneme strings, checked before
/// the lexicon and embedded fallback.
pub fn english_to_phonemes_with_overrides(
    text: &str,
    overrides: &HashMap<String, String>,
) -> Result<String, G2pError> {
    let g2p = G2P::new().with_overrides(overrides.clone());
    g2p.convert(text)
}

/// Post-process espeak-ng IPA output into Kokoro phoneme format.
///
/// Kept for backward compatibility. New code should use `english_to_phonemes()`.
pub fn espeak_ipa_to_kokoro(ipa: &str) -> String {
    let mut s = ipa.to_string();

    s = s.replace("dʒ", "ʤ");
    s = s.replace("tʃ", "ʧ");
    s = s.replace("ɜːɹ", "ɜɹ");
    s = s.replace("ɜː", "ɜɹ");
    s = s.replace("aɪ", "I");
    s = s.replace("aʊ", "W");
    s = s.replace("eɪ", "A");
    s = s.replace("oʊ", "O");
    s = s.replace("ɔɪ", "Y");
    s = s.replace('ː', "");
    s = s.replace('ɾ', "T");

    s
}

/// Split text into chunks whose phoneme representations fit within the model's
/// 510-character context limit.
pub fn text_to_phoneme_chunks(text: &str) -> Result<Vec<String>, G2pError> {
    const MAX_PHONEME_LEN: usize = 500;

    let mut chunks = Vec::new();

    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        let phonemes = english_to_phonemes(paragraph)?;
        if phonemes.len() <= MAX_PHONEME_LEN {
            chunks.push(phonemes);
            continue;
        }

        let sentences = split_sentences(paragraph);
        let mut current_phonemes = String::new();

        for sentence in &sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sent_phonemes = english_to_phonemes(sentence)?;

            if current_phonemes.is_empty() {
                current_phonemes = sent_phonemes;
            } else if current_phonemes.len() + 1 + sent_phonemes.len() <= MAX_PHONEME_LEN {
                current_phonemes.push(' ');
                current_phonemes.push_str(&sent_phonemes);
            } else {
                chunks.push(current_phonemes);
                current_phonemes = sent_phonemes;
            }
        }

        if !current_phonemes.is_empty() {
            chunks.push(current_phonemes);
        }
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    Ok(chunks)
}

/// Split text into chunks whose phoneme representations fit within the model's
/// 510-character context limit, with custom word-to-phoneme overrides.
///
/// Overrides map lowercase words to phoneme strings, checked before
/// the lexicon and embedded fallback.
pub fn text_to_phoneme_chunks_with_overrides(
    text: &str,
    overrides: &HashMap<String, String>,
) -> Result<Vec<String>, G2pError> {
    const MAX_PHONEME_LEN: usize = 500;

    let mut chunks = Vec::new();

    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        let phonemes = english_to_phonemes_with_overrides(paragraph, overrides)?;
        if phonemes.len() <= MAX_PHONEME_LEN {
            chunks.push(phonemes);
            continue;
        }

        let sentences = split_sentences(paragraph);
        let mut current_phonemes = String::new();

        for sentence in &sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sent_phonemes = english_to_phonemes_with_overrides(sentence, overrides)?;

            if current_phonemes.is_empty() {
                current_phonemes = sent_phonemes;
            } else if current_phonemes.len() + 1 + sent_phonemes.len() <= MAX_PHONEME_LEN {
                current_phonemes.push(' ');
                current_phonemes.push_str(&sent_phonemes);
            } else {
                chunks.push(current_phonemes);
                current_phonemes = sent_phonemes;
            }
        }

        if !current_phonemes.is_empty() {
            chunks.push(current_phonemes);
        }
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    Ok(chunks)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            sentences.push(current.clone());
            current.clear();
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affricate_conversion() {
        assert_eq!(espeak_ipa_to_kokoro("dʒʌmp"), "ʤʌmp");
        assert_eq!(espeak_ipa_to_kokoro("tʃɪp"), "ʧɪp");
    }

    #[test]
    fn test_diphthong_collapse() {
        assert_eq!(espeak_ipa_to_kokoro("haɪ"), "hI");
        assert_eq!(espeak_ipa_to_kokoro("naʊ"), "nW");
        assert_eq!(espeak_ipa_to_kokoro("deɪ"), "dA");
        assert_eq!(espeak_ipa_to_kokoro("goʊ"), "gO");
        assert_eq!(espeak_ipa_to_kokoro("bɔɪ"), "bY");
    }

    #[test]
    fn test_nurse_vowel() {
        assert_eq!(espeak_ipa_to_kokoro("wɜːɹld"), "wɜɹld");
        assert_eq!(espeak_ipa_to_kokoro("bɜːd"), "bɜɹd");
    }

    #[test]
    fn test_length_mark_removal() {
        assert_eq!(espeak_ipa_to_kokoro("siː"), "si");
        assert_eq!(espeak_ipa_to_kokoro("fuːd"), "fud");
    }

    #[test]
    fn test_flap_to_t() {
        assert_eq!(espeak_ipa_to_kokoro("wɑɾɚ"), "wɑTɚ");
    }

    #[test]
    fn test_full_espeak_output() {
        let input = "həlˈoʊ wˈɜːld";
        let expected = "həlˈO wˈɜɹld";
        assert_eq!(espeak_ipa_to_kokoro(input), expected);
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(
            sentences,
            vec!["Hello world.", " How are you?", " I'm fine!"]
        );
    }

    #[test]
    fn test_g2p_convert_hello() {
        let g2p = G2P::new();
        let result = g2p.convert("hello").unwrap();
        assert!(!result.is_empty());
        assert!(
            result.contains('O') || result.contains('o'),
            "Expected phonemes for 'hello', got: {}",
            result
        );
    }

    #[test]
    fn test_g2p_convert_sentence() {
        let g2p = G2P::new();
        let result = g2p.convert("Hello world").unwrap();
        assert!(!result.is_empty());
        assert!(
            result.contains(' '),
            "Expected space between words in: {}",
            result
        );
    }

    #[test]
    fn test_g2p_hyphen_compound_preserves_word_boundary() {
        let g2p = G2P::new();
        let result = g2p.convert("cat-dog").unwrap();
        assert!(
            result.contains(' '),
            "Expected a phoneme-space boundary for hyphen compound, got: {result}"
        );
    }

    #[test]
    fn test_g2p_underscore_compound_preserves_word_boundary() {
        let g2p = G2P::new();
        let result = g2p.convert("cat_dog").unwrap();
        assert!(
            result.contains(' '),
            "Expected a phoneme-space boundary for underscore compound, got: {result}"
        );
    }

    #[test]
    fn test_g2p_hyphen_fallback_keeps_resolved_subtokens_without_espeak() {
        let g2p = G2P::with_config(G2PConfig {
            espeak_path: "/definitely/missing/espeak-ng".to_string(),
        });
        let result = g2p.convert("cat-unpronounceablexyz-dog").unwrap();
        assert!(
            result.contains(' '),
            "Expected resolved neighboring words to remain separated, got: {result}"
        );
        assert!(
            result.contains("d"),
            "Expected resolved trailing word to remain present, got: {result}"
        );
    }

    #[test]
    fn test_g2p_convert_the_context() {
        let g2p = G2P::new();
        let result = g2p.convert("the apple").unwrap();
        assert!(
            result.contains("ði"),
            "Expected 'ði' (the before vowel) in: {}",
            result
        );
    }

    #[test]
    fn test_g2p_convert_number() {
        let g2p = G2P::new();
        let result = g2p.convert("42").unwrap();
        assert!(!result.is_empty(), "Should produce phonemes for numbers");
    }

    #[test]
    fn test_english_to_phonemes_api() {
        let result = english_to_phonemes("hello world");
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_oov_fallback_does_not_require_espeak() {
        let g2p = G2P::with_config(G2PConfig {
            espeak_path: "/definitely/missing/espeak-ng".into(),
        });
        let result = g2p.convert("neologismxyz").unwrap();
        assert!(
            !result.trim().is_empty(),
            "OOV fallback should produce phonemes without espeak-ng"
        );
        assert!(
            result.contains('\u{02C8}'),
            "OOV fallback should assign primary stress: {result}"
        );
    }

    // -- Punctuation preservation tests --------------------------------------

    #[test]
    fn test_period_preserved() {
        let result = english_to_phonemes("Hello.").unwrap();
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_comma_preserved() {
        let result = english_to_phonemes("Hello, world.").unwrap();
        assert!(
            result.contains(','),
            "Comma should appear in phonemes: {result}"
        );
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_question_mark_preserved() {
        let result = english_to_phonemes("Hello?").unwrap();
        assert!(
            result.contains('?'),
            "Question mark should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_exclamation_preserved() {
        let result = english_to_phonemes("Hello!").unwrap();
        assert!(
            result.contains('!'),
            "Exclamation mark should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_two_sentences_have_period_between() {
        let result = english_to_phonemes("Hello. World.").unwrap();
        // Should have at least one period (ideally two) in the phoneme output
        let period_count = result.chars().filter(|c| *c == '.').count();
        assert!(
            period_count >= 1,
            "Expected period(s) between sentences, got: {result}"
        );
    }

    #[test]
    fn test_mixed_punctuation() {
        let result = english_to_phonemes("Wait! What? Really.").unwrap();
        assert!(
            result.contains('!'),
            "Exclamation should appear in phonemes: {result}"
        );
        assert!(
            result.contains('?'),
            "Question mark should appear in phonemes: {result}"
        );
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_semicolon_preserved() {
        let result = english_to_phonemes("Hello; world.").unwrap();
        assert!(
            result.contains(';'),
            "Semicolon should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_builtin_overrides() {
        let g2p = G2P::new();
        let pronounces = |text: &str| g2p.convert(text).unwrap().trim().to_string();
        assert_eq!(pronounces("ACL"), "ˈA sˈi ˈɛl");
        assert_eq!(pronounces("API"), "ˈA pˈi ˌI");
        assert_eq!(pronounces("APIs"), "ˈA pˈi ˈIz");
        assert_eq!(pronounces("anywidget"), "ˈɛni wˌɪʤət");
        assert_eq!(pronounces("Automerge"), "ˈɔTO mˈɜɹʤ");
        assert_eq!(pronounces("automunge"), "ˈɔTO mˈʌnʤ");
        assert_eq!(pronounces("AWS"), "ˈA dˈʌbᵊlju ˈɛs");
        assert_eq!(pronounces("BiLSTM"), "bˈI ˈɛl ˈɛs tˈi ˈɛm");
        assert_eq!(pronounces("BYOC"), "bˈi wˈI ˈO sˈi");
        assert_eq!(pronounces("CLI"), "sˈi ˈɛl ˌI");
        assert_eq!(pronounces("CLIs"), "sˈi ˈɛl ˈIz");
        assert_eq!(pronounces("Cloudflare"), "klˈWd flˈɛɹ");
        assert_eq!(pronounces("CoreAudio"), "kˈɔɹ ˈɔdiO");
        assert_eq!(pronounces("CoreML"), "kˈɔɹ ˈɛm ˈɛl");
        assert_eq!(pronounces("CPAL"), "sˈi pˈi ˈA ˈɛl");
        assert_eq!(pronounces("CRDT"), "sˈi ˈɑɹ dˈi tˈi");
        assert_eq!(pronounces("CRDTs"), "sˈi ˈɑɹ dˈi tˈiz");
        assert_eq!(pronounces("CSR"), "sˈi ˈɛs ˈɑɹ");
        assert_eq!(pronounces("CSS"), "sˈi ˈɛs ˈɛs");
        assert_eq!(pronounces("CUDA"), "kˈudə");
        assert_eq!(pronounces("D1"), "dˈi wˈʌn");
        assert_eq!(pronounces("Deno"), "dˈinO");
        assert_eq!(pronounces("demos"), "dˈɛmOz");
        assert_eq!(pronounces("demo"), "dˈɛmO");
        assert_eq!(pronounces("TODO"), "tˈudu");
        assert_eq!(pronounces("demuxing"), "dˌimˈʌksɪŋ");
        assert_eq!(pronounces("demux"), "dˌimˈʌks");
        assert_eq!(pronounces("demultiplexing"), "dˌimˈʌltɪplɛksɪŋ");
        assert_eq!(pronounces("demultiplex"), "dˌimˈʌltɪplɛks");
        assert_eq!(pronounces("DFT"), "dˈi ˈɛf tˈi");
        assert_eq!(pronounces("DTLS"), "dˈi tˈi ˈɛl ˈɛs");
        assert_eq!(pronounces("DuckDB"), "dˈʌk dˈi bˈi");
        assert_eq!(pronounces("esbuild"), "ˈi ˈɛs bˌɪld");
        assert_eq!(pronounces("ESLint"), "ˈi ˈɛs lˌɪnt");
        assert_eq!(pronounces("espeak"), "ˈi spˌik");
        assert_eq!(pronounces("FastAPI"), "fˈæst ˈA pˈi ˌI");
        assert_eq!(pronounces("FLAC"), "flˈæk");
        assert_eq!(pronounces("GGUF"), "ʤˈi ʤˈi jˈu ˈɛf");
        assert_eq!(pronounces("gRPC"), "ʤˈi ˈɑɹ pˈi sˈi");
        assert_eq!(pronounces("GPT"), "ʤˈi pˈi tˈi");
        assert_eq!(pronounces("ChatGPT"), "ʧˈæt ʤˈi pˈi tˈi");
        assert_eq!(pronounces("HTTP"), "ˈAʧ tˈi tˈi pˈi");
        assert_eq!(pronounces("HTTPS"), "ˈAʧ tˈi tˈi pˈi ˈɛs");
        assert_eq!(pronounces("HTML"), "ˈAʧ tˈi ˈɛm ˈɛl");
        assert_eq!(
            pronounces("HTMLIFrameElement"),
            "ˈAʧ tˈi ˈɛm ˈɛl ˌI fɹˌAm ˈɛləmənt"
        );
        assert_eq!(pronounces("ID"), "ˈI dˈi");
        assert_eq!(pronounces("IDs"), "ˈI dˈiz");
        assert_eq!(pronounces("IDB"), "ˌI dˈi bˈi");
        assert_eq!(pronounces("iframe"), "ˌI fɹˌAm");
        assert_eq!(pronounces("iOS"), "ˈI ˈO ˈɛs");
        assert_eq!(pronounces("IndexedDB"), "ˈɪndɛkst dˈi bˈi");
        assert_eq!(pronounces("IPC"), "ˌI pˈi sˈi");
        assert_eq!(pronounces("ipykernel"), "ˈI pˈI kˌɜɹnᵊl");
        assert_eq!(pronounces("IPython"), "ˌI pˈIθˌɑn");
        assert_eq!(pronounces("ipywidgets"), "ˌI pˈI wˌɪʤəts");
        assert_eq!(pronounces("isort"), "ˈI sˌɔɹt");
        assert_eq!(pronounces("iSTFT"), "ˈI ˈɛs tˈi ˈɛf tˈi");
        assert_eq!(pronounces("JAX"), "ʤˈæks");
        assert_eq!(pronounces("JSON"), "ʤˌA sˈæhn");
        assert_eq!(pronounces("JSONRPC"), "ʤˌA sˈæhn ˈɑɹ pˈi sˈi");
        assert_eq!(pronounces("JS"), "ʤˈA ˈɛs");
        assert_eq!(pronounces("JSX"), "ʤˈA ˈɛs ˈɛks");
        assert_eq!(pronounces("Jupyter"), "ʤˈupɪTəɹ");
        assert_eq!(pronounces("JWT"), "ʤˈA dˈʌbᵊlju tˈi");
        assert_eq!(pronounces("JWTs"), "ʤˈA dˈʌbᵊlju tˈiz");
        assert_eq!(pronounces("KaTeX"), "kˈA tˌɛk");
        assert_eq!(pronounces("kernelspec"), "kˈɜɹnᵊl spˌɛk");
        assert_eq!(pronounces("Kokoro"), "kˈOkəɹO");
        assert_eq!(pronounces("Kokoro-82M"), "kˈOkəɹO ˈATi tˈu ˈɛm");
        assert_eq!(pronounces("Kubernetes"), "kˌubəɹnˈɛtiz");
        assert_eq!(pronounces("kubectl"), "kjˈub kˈʌdᵊl");
        assert_eq!(pronounces("LaTeX"), "lˈA tˌɛk");
        assert_eq!(pronounces("LFS"), "ˈɛl ˈɛf ˈɛs");
        assert_eq!(pronounces("LSTM"), "ˈɛl ˈɛs tˈi ˈɛm");
        assert_eq!(pronounces("macOS"), "mˈæk ˈO ˈɛs");
        assert_eq!(pronounces("Matplotlib"), "mˈæt plˌɑt lˌɪb");
        assert_eq!(pronounces("MathJax"), "mˈæθ ʤˌæks");
        assert_eq!(pronounces("MCP"), "ˈɛm sˈi pˈi");
        assert_eq!(pronounces("MCPs"), "ˈɛm sˈi pˈiz");
        assert_eq!(pronounces("MDX"), "ˈɛm dˈi ˈɛks");
        assert_eq!(pronounces("micropip"), "mˈIkɹO pˌɪp");
        assert_eq!(pronounces("MIMEBundle"), "mˈIm bˌʌndᵊl");
        assert_eq!(pronounces("MLX"), "ˈɛm ˈɛl ˈɛks");
        assert_eq!(pronounces("mmap"), "ˈɛm mˌæp");
        assert_eq!(pronounces("MSW"), "ˈɛm ˈɛs dˈʌbᵊlju");
        assert_eq!(pronounces("mypy"), "mˈI pˌI");
        assert_eq!(pronounces("nbconvert"), "ˈɛn bˈi kˌɑnvɜɹt");
        assert_eq!(pronounces("nbformat"), "ˈɛn bˈi fˌɔɹmæt");
        assert_eq!(pronounces("Neuphonic"), "nˈu fˌɑnɪk");
        assert_eq!(pronounces("NeuTTS"), "nˈu tˈi tˈi ˈɛs");
        assert_eq!(pronounces("Next.js"), "nˈɛkst ʤˈA ˈɛs");
        assert_eq!(pronounces("NextJS"), "nˈɛkst ʤˈA ˈɛs");
        assert_eq!(pronounces("nginx"), "ˈɛnʤən ˌɛks");
        assert_eq!(pronounces("Node.js"), "nˈOd ʤˈA ˈɛs");
        assert_eq!(pronounces("NodeJS"), "nˈOd ʤˈA ˈɛs");
        assert_eq!(pronounces("nteract"), "ˈɛntəɹˌækt");
        assert_eq!(pronounces("nteract-dev"), "ˈɛntəɹˌækt dˈɛv");
        assert_eq!(pronounces("NumPy"), "nˈʌm pˌI");
        assert_eq!(pronounces("OAuth"), "ˌO ˈɔθ");
        assert_eq!(pronounces("OGG"), "ˈɑɡ");
        assert_eq!(pronounces("OIDC"), "ˈO ˌI dˈi sˈi");
        assert_eq!(pronounces("ONNX"), "ˈɑnɪks");
        assert_eq!(pronounces("openai-codex"), "ˌOpᵊn ˈAˌI kˈOdˌɛks");
        assert_eq!(pronounces("OpenAPI"), "ˈOpᵊn ˈA pˈi ˌI");
        assert_eq!(pronounces("OPFS"), "ˈO pˈi ˈɛf ˈɛs");
        assert_eq!(pronounces("Outerbounds"), "ˈWTəɹ bˈWndz");
        assert_eq!(pronounces("outputIdChanges"), "ˈWtpˌʊt ˌI dˌi ʧˈAnʤᵻz");
        assert_eq!(pronounces("pnpm"), "pˈi ˈɛn pˈi ˈɛm");
        assert_eq!(pronounces("Postgres"), "pˈOstɡɹɛs");
        assert_eq!(pronounces("PostgreSQL"), "pˈOst ɡɹˈɛs kjˈu ˈɛl");
        assert_eq!(pronounces("PCM"), "pˈi sˈi ˈɛm");
        assert_eq!(pronounces("Protobuf"), "pɹˈOTO bˌʌf");
        assert_eq!(pronounces("PWA"), "pˈi dˈʌbᵊlju ˈA");
        assert_eq!(pronounces("PUT_BLOB"), "pˌʊt blˈɑb");
        assert_eq!(pronounces("PutBlob"), "pˌʊt blˈɑb");
        assert_eq!(pronounces("PyArrow"), "pˈI ˈɛɹO");
        assert_eq!(pronounces("Pyodide"), "pˈI ə dˌId");
        assert_eq!(pronounces("PyPI"), "pˈI pˈi ˌI");
        assert_eq!(pronounces("PyO3"), "pˈI ˈO θɹˈi");
        assert_eq!(pronounces("pytest"), "pˈI tˌɛst");
        assert_eq!(pronounces("QQBot"), "kjˈu kjˈu bˌɑt");
        assert_eq!(pronounces("R2"), "ˈɑɹ tˈu");
        assert_eq!(pronounces("PyTorch"), "pˈI tˌɔɹʧ");
        assert_eq!(pronounces("vitest"), "vˈi tˌɛst");
        assert_eq!(pronounces("tsconfig"), "tˈi ˈɛs kˌɑnfˈɪɡ");
        assert_eq!(pronounces("ipynb"), "nˈOtbˌʊk fˈIl");
        assert_eq!(pronounces("PR"), "pˈi ˈɑɹ");
        assert_eq!(pronounces("PRs"), "pˈi ˈɑɹz");
        assert_eq!(pronounces("RPC"), "ˈɑɹ pˈi sˈi");
        assert_eq!(pronounces("RPCs"), "ˈɑɹ pˈi sˈiz");
        assert_eq!(pronounces("Rodio"), "ɹˈOdiˌO");
        assert_eq!(pronounces("RTCP"), "ˈɑɹ tˈi sˈi pˈi");
        assert_eq!(pronounces("RTP"), "ˈɑɹ tˈi pˈi");
        assert_eq!(pronounces("repr-llm"), "ɹˈɛpəɹ ˈɛl ˈɛl ˈɛm");
        assert_eq!(pronounces("runt-nightly"), "ɹˈʌnt nˈItli");
        assert_eq!(pronounces("runtimed"), "ɹˈʌntIm dˈi");
        assert_eq!(pronounces("runtimed-wasm"), "ɹˈʌntIm dˈi wˈæzəm");
        assert_eq!(pronounces("runtime-peer"), "ɹˈʌntIm pˈɪɹ");
        assert_eq!(pronounces("runtime_peer"), "ɹˈʌntIm pˈɪɹ");
        assert_eq!(pronounces("S3"), "ˈɛs θɹˈi");
        assert_eq!(pronounces("safetensors"), "sˈAf tˌɛnsəɹz");
        assert_eq!(pronounces("scikit-learn"), "sˈɪkɪt lˌɜɹn");
        assert_eq!(pronounces("sklearn"), "ˈɛs kˈA lˌɜɹn");
        assert_eq!(pronounces("SciPy"), "sˈI pˌI");
        assert_eq!(pronounces("SDP"), "ˈɛs dˈi pˈi");
        assert_eq!(pronounces("Serde"), "sˈɜɹdˌi");
        assert_eq!(pronounces("SQLite"), "ˌɛs kjˈu ˌɛl lˈIt");
        assert_eq!(pronounces("SCSS"), "ˈɛs sˈi ˈɛs ˈɛs");
        assert_eq!(pronounces("SRTP"), "ˈɛs ˈɑɹ tˈi pˈi");
        assert_eq!(pronounces("SSR"), "ˈɛs ˈɛs ˈɑɹ");
        assert_eq!(pronounces("STFT"), "ˈɛs tˈi ˈɛf tˈi");
        assert_eq!(pronounces("STT"), "ˈɛs tˈi tˈi");
        assert_eq!(pronounces("Supabase"), "sˈupə bˌAs");
        assert_eq!(pronounces("SWC"), "ˈɛs dˈʌbᵊlju sˈi");
        assert_eq!(pronounces("TailwindCSS"), "tˈAl wˈɪnd sˈi ˈɛs ˈɛs");
        assert_eq!(pronounces("Tauri"), "tˈW ɹˌi");
        assert_eq!(pronounces("TeX"), "tˈɛk");
        assert_eq!(pronounces("tokenizer"), "tˈOkᵊn ˌIzəɹ");
        assert_eq!(pronounces("tokenizers"), "tˈOkᵊn ˌIzəɹz");
        assert_eq!(pronounces("TOML"), "tˈɑmᵊl");
        assert_eq!(pronounces("Tokio"), "tˈOkiˌO");
        assert_eq!(pronounces("TTS"), "tˈi tˈi ˈɛs");
        assert_eq!(pronounces("TSX"), "tˈi ˈɛs ˈɛks");
        assert_eq!(pronounces("TypeScript"), "tˈIp skɹˌɪpt");
        assert_eq!(pronounces("UI"), "jˈu ˈI");
        assert_eq!(pronounces("ULID"), "jˈu lˌɪd");
        assert_eq!(pronounces("URL"), "jˈu ˈɑɹ ˈɛl");
        assert_eq!(pronounces("URLs"), "jˈu ˈɑɹ ˈɛlz");
        assert_eq!(pronounces("URI"), "jˈu ˈɑɹ ˌI");
        assert_eq!(pronounces("URIs"), "jˈu ˈɑɹ ˈIz");
        assert_eq!(pronounces("Utf8View"), "jˈu tˈi ˈɛf ˈAt vjˈu");
        assert_eq!(pronounces("UX"), "jˈu ˈɛks");
        assert_eq!(pronounces("UUID"), "jˈu jˈu ˌI dˈi");
        assert_eq!(pronounces("UUIDs"), "jˈu jˈu ˌI dˈiz");
        assert_eq!(pronounces("VAD"), "vˈi ˈA dˈi");
        assert_eq!(pronounces("VADs"), "vˈi ˈA dˈiz");
        assert_eq!(pronounces("Vite"), "vˈit");
        assert_eq!(pronounces("Vitest"), "vˈi tˌɛst");
        assert_eq!(pronounces("Vue"), "vjˈu");
        assert_eq!(pronounces("VSCode"), "vˈi ˈɛs kˈOd");
        assert_eq!(pronounces("WAV"), "wˈAv");
        assert_eq!(pronounces("WAVs"), "wˈAvz");
        assert_eq!(pronounces("WASM"), "wˈæzəm");
        assert_eq!(pronounces("WebRTC"), "wˈɛb ˈɑɹ tˈi sˈi");
        assert_eq!(pronounces("WebSocket"), "wˈɛb sˌɑkət");
        assert_eq!(pronounces("WebSockets"), "wˈɛb sˌɑkəts");
        assert_eq!(pronounces("WiFi"), "wˈI fˌI");
        assert_eq!(pronounces("xAI"), "ˈɛks ˈA ˌI");
        assert_eq!(pronounces("YAML"), "jˈæmᵊl");
        assert_eq!(pronounces("Yjs"), "wˈI ʤˈA ˈɛs");
        assert_eq!(pronounces("ZeroMQ"), "zˈɪɹO ˈɛm kjˈu");
        assert_eq!(pronounces("ZMQ"), "zˈi ˈɛm kjˈu");
    }

    #[test]
    fn test_nteract_identifier_phrases() {
        let g2p = G2P::new();
        let assert_contains = |text: &str, expected: &[&str]| {
            let result = g2p.convert(text).unwrap();
            for part in expected {
                assert!(
                    result.contains(part),
                    "expected {text:?} to contain {part:?}, got {result:?}"
                );
            }
        };

        assert_contains(
            "runtime_peer writes RuntimeStateDoc and CommsDoc",
            &["ɹˈʌntIm pˈɪɹ", "ɹˈʌntIm stˈAt dˌɑk", "kˈɑmz dˌɑk"],
        );
        assert_contains(
            "ACL checks gate PUT_BLOB in Cloudflare D1 and R2",
            &[
                "ˈA sˈi ˈɛl",
                "pˌʊt blˈɑb",
                "klˈWd flˈɛɹ",
                "dˈi wˈʌn",
                "ˈɑɹ tˈu",
            ],
        );
        assert_contains(
            "nteract.dx.blob uses SyncEngine and CommBridgeManager",
            &[
                "ˈɛntəɹˌækt dˈi ˈɛks blˌɑb",
                "sˌɪŋk ˈɛnʤən",
                "kˌɑm bɹˈɪʤ mˈænɪʤəɹ",
            ],
        );
        assert_contains(
            "runt-nightly starts runtimed-wasm for nteract-dev",
            &["ɹˈʌnt nˈItli", "ɹˈʌntIm dˈi wˈæzəm", "ˈɛntəɹˌækt dˈɛv"],
        );
        assert_contains(
            "Arrow IPC uses Utf8View in Sift",
            &["ˈɛɹO", "ˌI pˈi sˈi", "jˈu tˈi ˈɛf ˈAt vjˈu", "sˈɪft"],
        );
        assert_contains(
            "HTMLElement reads URLSearchParams and cell_id in NotebookUI",
            &[
                "ˈAʧ tˈi ˈɛm ˈɛl ˌɛləmənt",
                "jˈu ˈɑɹ ˈɛl sˌɜɹʧ pˈɑɹæms",
                "sˌɛl ˈI dˈi",
                "nˈOtbˌʊk jˌu ˌI",
            ],
        );
    }

    // -- camelCase tests ------------------------------------------------------

    #[test]
    fn test_camel_case_spaced_phonemes() {
        let g2p = G2P::new();

        // Two-part camelCase
        let result = g2p.convert("useEffect").unwrap();
        assert!(
            result.contains(' '),
            "camelCase should produce space-separated phonemes: {result}"
        );

        // Three-part camelCase (using common dictionary words)
        let result = g2p.convert("getInputValue").unwrap();
        let spaces = result.chars().filter(|c| *c == ' ').count();
        assert!(
            spaces >= 2,
            "Three-part camelCase should have 2+ spaces: {result}"
        );

        // Single word should not gain a space
        let result = g2p.convert("hello").unwrap();
        assert!(
            !result.contains(' '),
            "Single word should not have spaces: {result}"
        );
    }
}
