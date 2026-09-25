"""
Add 2 more real-code SAFE/VULNERABLE pairs for Bump Seed Canonicalization and
Type Confusion, sourced from coral-xyz/sealevel-attacks -- the reference
repo (maintained by the Anchor/Coral team) cataloguing common Solana
exploits with minimal insecure/recommended program pairs. This is a
different, complementary source from add_bump_seed_type_confusion_examples.py
(real deployed SPL programs): here the code is Anchor-framework style
(#[program]/#[account(...)] macros) rather than raw native
processor.rs/next_account_info style, which adds coding-style diversity the
training set didn't have before.

Source (both "direct extraction", trimmed of licensing boilerplate only):
  https://github.com/coral-xyz/sealevel-attacks/blob/master/programs/7-bump-seed-canonicalization/{insecure,recommended}/src/lib.rs
  https://github.com/coral-xyz/sealevel-attacks/blob/master/programs/3-type-cosplay/{insecure,recommended}/src/lib.rs
"""

import json
import os

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
TRAIN_PATH = os.path.join(REPO_DIR, "data", "training", "dataset_think_format.jsonl")

DEVELOPER_PROMPT = """You are an expert smart contract security auditor specialized in the Solana blockchain and Rust.
Your task is to analyze Rust code precisely and systematically to identify security vulnerabilities.

### Required behavior:
- Always perform a structured, explicit reasoning phase first and put it inside the <think> block.
  - In <think>...</think> you must:
    - Summarize the contract's purpose and high-level architecture.
    - Inspect the logic block-by-block (or line-by-line for short snippets).
    - Note any suspicious patterns, missing authorization checks, unsafe operations, or other issues with evidence (point to the code lines/constructs).
    - Use concise, technical language and show the chain of reasoning (why you suspect an issue).

- After </think>, provide the final judgment in the <final>...</final> block using this exact structure:
  - A short summary sentence (one or two lines).
  - ### Vulnerability: <name or "Not Vulnerable">
  - ### Explanation: <concise cause and how it can be exploited>
  - ### Risk: <severity (Critical/High/Medium/Low) and short impact statement>

### Constraints:
- Do NOT include any extra commentary, greetings, or meta-text.
- If no vulnerability is found, explicitly write "### Vulnerability: Not Vulnerable" and "### Risk: No significant risk identified"."""

USER_TEMPLATE = """Please perform a detailed security analysis of the following Solana smart contract.
Carefully examine its logic, identify any potential vulnerabilities:

Contract Code:

{code}"""


def sample(id_, code, thinking, content):
    return {
        "id": id_,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(code=code)},
            {"role": "assistant", "thinking": thinking, "content": content},
        ],
    }


# ============================================================
# Bump Seed Canonicalization -- sealevel-attacks/7-bump-seed-canonicalization
# ============================================================

CODE_BUMP_VULN = """use anchor_lang::prelude::*;

#[program]
pub mod bump_seed_canonicalization_insecure {
    use super::*;

    pub fn set_value(ctx: Context<BumpSeed>, key: u64, new_value: u64, bump: u8) -> ProgramResult {
        let address =
            Pubkey::create_program_address(&[key.to_le_bytes().as_ref(), &[bump]], ctx.program_id)?;
        if address != ctx.accounts.data.key() {
            return Err(ProgramError::InvalidArgument);
        }

        ctx.accounts.data.value = new_value;

        Ok(())
    }
}

#[derive(Accounts)]
pub struct BumpSeed<'info> {
    data: Account<'info, Data>,
}

#[account]
pub struct Data {
    value: u64,
}"""

CODE_BUMP_SAFE = """use anchor_lang::prelude::*;

#[program]
pub mod bump_seed_canonicalization_recommended {
    use super::*;

    pub fn set_value(ctx: Context<BumpSeed>, key: u64, new_value: u64) -> ProgramResult {
        ctx.accounts.data.value = new_value;
        Ok(())
    }
}

#[derive(Accounts)]
#[instruction(key: u64)]
pub struct BumpSeed<'info> {
    // Anchor's seeds/bump constraint derives the address internally via
    // find_program_address, enforcing the canonical bump before the account
    // is even made available to the handler -- no bump is ever accepted as
    // a raw instruction argument.
    #[account(seeds = [key.to_le_bytes().as_ref()], bump)]
    data: Account<'info, Data>,
}

#[account]
pub struct Data {
    value: u64,
}"""

THINKING_BUMP_VULN = """Analyzing the Solana smart contract function `set_value`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `set_value`:
   - `bump` is taken directly as an instruction argument and passed straight into `Pubkey::create_program_address(&[key.to_le_bytes().as_ref(), &[bump]], ctx.program_id)`. The only check performed is that the resulting address equals `ctx.accounts.data.key()` -- there is no verification that `bump` is the unique canonical bump `find_program_address` would have returned for this `key`.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate PDA derivation and bump seed handling. Since `create_program_address` accepts any bump that produces a valid off-curve point, an attacker can precompute a different (non-canonical) bump for a `key` of their choosing that collides with an account they control, then pass that `key`/`bump`/account combination to have the program treat their own account as the legitimate PDA for that key.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_BUMP_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V9 - Bump Seed Canonicalization
### Explanation: The bump seed is accepted directly as an instruction argument and used with create_program_address instead of being derived canonically via find_program_address. An attacker can supply a non-canonical bump/key pair that maps to an account they control, spoofing the intended PDA.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_BUMP_SAFE = """Analyzing the Solana smart contract function `set_value`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for PDA derivation and bump seed handling.

2. **Expected security checks for V9 (Bump Seed)**:
   - Check 1: use of find_program_address (returns canonical bump)
   - Check 2: bump seed stored and verified on subsequent calls
   - Check 3: no raw create_program_address without bump validation

3. **Code inspection**: Examining the implementation of `set_value`:
   - The `data` account is constrained with `#[account(seeds = [key.to_le_bytes().as_ref()], bump)]`. This Anchor constraint derives the PDA internally via `find_program_address` and rejects the transaction before the handler body even runs if the account does not match the canonical address. No bump is accepted as a raw instruction argument.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V9 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles PDA derivation and bump seed canonicalization appropriately."""

CONTENT_BUMP_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The data account's PDA is derived and enforced via Anchor's seeds/bump constraint, which uses find_program_address internally and rejects any account that isn't the canonical PDA before the handler runs.
### Risk: No significant risk identified."""


# ============================================================
# Type Confusion (type cosplay) -- sealevel-attacks/3-type-cosplay
# ============================================================

CODE_TYPECONF_VULN = """use anchor_lang::prelude::*;
use borsh::{BorshDeserialize, BorshSerialize};

#[program]
pub mod type_cosplay_insecure {
    use super::*;

    pub fn update_user(ctx: Context<UpdateUser>) -> ProgramResult {
        let user = User::try_from_slice(&ctx.accounts.user.data.borrow()).unwrap();
        if ctx.accounts.user.owner != ctx.program_id {
            return Err(ProgramError::IllegalOwner);
        }
        if user.authority != ctx.accounts.authority.key() {
            return Err(ProgramError::InvalidAccountData);
        }
        msg!("GM {}", user.authority);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct UpdateUser<'info> {
    user: AccountInfo<'info>,
    authority: Signer<'info>,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct User {
    authority: Pubkey,
}

#[derive(BorshSerialize, BorshDeserialize)]
pub struct Metadata {
    account: Pubkey,
}"""

CODE_TYPECONF_SAFE = """use anchor_lang::prelude::*;
use borsh::{BorshDeserialize, BorshSerialize};

#[program]
pub mod type_cosplay_recommended {
    use super::*;

    pub fn update_user(ctx: Context<UpdateUser>) -> ProgramResult {
        msg!("GM {}", ctx.accounts.user.authority);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct UpdateUser<'info> {
    // Account<'info, User> makes Anchor check an 8-byte type discriminator
    // (derived from the struct name) at the start of the account data before
    // deserializing, and has_one = authority cross-validates the field.
    #[account(has_one = authority)]
    user: Account<'info, User>,
    authority: Signer<'info>,
}

#[account]
pub struct User {
    authority: Pubkey,
}

#[account]
pub struct Metadata {
    account: Pubkey,
}"""

THINKING_TYPECONF_VULN = """Analyzing the Solana smart contract function `update_user`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `update_user`:
   - `user` is a raw `AccountInfo`, deserialized with `User::try_from_slice(&ctx.accounts.user.data.borrow())`. The code checks `owner == ctx.program_id` and that the deserialized `authority` field matches the signer, but `User` and `Metadata` share the exact same underlying encoding (both are a single `Pubkey`), and neither struct carries a discriminator tag.

4. **Finding**: The code fails to implement the required security checks. Specifically, the vulnerability stems from missing or inadequate input validation and account type verification. An attacker can pass a `Metadata` account (or any other program-owned account whose first 32 bytes happen to equal their own key) where a `User` account is expected; `try_from_slice` will happily reinterpret its bytes as a `User`, and if the account's `account` field (interpreted as `authority`) matches the signer, the forged type is accepted.

5. **Severity assessment**: HIGH - This vulnerability can be directly exploited by a malicious actor."""

CONTENT_TYPECONF_VULN = """- The contract contains a security vulnerability due to missing or inadequate protections.
### Vulnerability: V4 - Type Confusion (Input Validation)
### Explanation: The user account is deserialized with a raw try_from_slice into a User struct with no discriminator/type tag, and User shares its byte layout with the unrelated Metadata struct. An attacker can substitute a different account type with a matching layout and have it accepted as a User.
### Risk: HIGH - This vulnerability can be exploited to compromise the contract's security and potentially cause fund loss."""

THINKING_TYPECONF_SAFE = """Analyzing the Solana smart contract function `update_user`.

1. **Function overview**: This function is part of a Solana program. I need to examine it for input validation and account type verification.

2. **Expected security checks for V4 (Type Confusion)**:
   - Check 1: account type/discriminator validation
   - Check 2: account ownership verification (e.g., account.owner == expected_program)
   - Check 3: data deserialization with proper type checks

3. **Code inspection**: Examining the implementation of `update_user`:
   - `user` is typed as `Account<'info, User>`, which makes Anchor check an 8-byte discriminator derived from the struct name at deserialization time and reject the transaction if the account is actually a `Metadata` (or any other type) before the handler body runs. `has_one = authority` further cross-validates the field against the signer.

4. **Assessment**: The code correctly implements the required security measures. The expected checks for V4 are present and properly enforced.

5. **Conclusion**: No vulnerability detected. The contract handles input validation and account type verification appropriately."""

CONTENT_TYPECONF_SAFE = """- The contract implements proper security measures against the analyzed vulnerability class.
### Vulnerability: Not Vulnerable
### Explanation: The user account is loaded as Account<'info, User>, so Anchor validates its type discriminator before use and has_one = authority cross-checks the authority field against the signer.
### Risk: No significant risk identified."""


def main():
    train_samples = [
        sample("bump_seed_vulnerable_213_sealevel_attacks", CODE_BUMP_VULN, THINKING_BUMP_VULN, CONTENT_BUMP_VULN),
        sample("bump_seed_safe_213_sealevel_attacks", CODE_BUMP_SAFE, THINKING_BUMP_SAFE, CONTENT_BUMP_SAFE),
        sample("type_confusion_vulnerable_213_sealevel_attacks", CODE_TYPECONF_VULN, THINKING_TYPECONF_VULN, CONTENT_TYPECONF_VULN),
        sample("type_confusion_safe_213_sealevel_attacks", CODE_TYPECONF_SAFE, THINKING_TYPECONF_SAFE, CONTENT_TYPECONF_SAFE),
    ]

    with open(TRAIN_PATH, "a", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Appended {len(train_samples)} samples to {TRAIN_PATH}")


if __name__ == "__main__":
    main()
