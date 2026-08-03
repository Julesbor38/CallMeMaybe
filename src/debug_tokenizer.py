from llm_sdk import Small_LLM_Model


def main() -> None:
    llm = Small_LLM_Model()

    functions = [
        "fn_add_numbers",
        "fn_greet",
        "fn_reverse_string",
        "fn_get_square_root",
        "fn_substitute_string_with_regex",
    ]

    print("=" * 80)
    print("TOKENIZATION")
    print("=" * 80)

    for function in functions:
        ids = [int(token) for token in llm.encode(function)[0]]

        print(function)
        print(f"ids    : {ids}")

        print("tokens :")
        for token in ids:
            piece = llm.decode([token])
            print(f"  {token:>6} -> {repr(piece)}")

        print("-" * 80)


if __name__ == "__main__":
    main()
