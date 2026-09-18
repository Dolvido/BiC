"""Keep installation diagnostics usable before optional ML imports succeed."""


def main(argv=None):
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "english-loop":
        from .dialogue_learning import main as english_main
        return english_main(arguments[1:])
    if arguments and arguments[0] == "learn-loop":
        from .learning_loop import main as learning_main
        return learning_main(arguments[1:])
    if arguments and arguments[0] in ("doctor", "launch"):
        from .launcher import main as launcher_main
        return launcher_main(arguments)
    from .cli import main as model_main
    return model_main(arguments)

if __name__ == "__main__":
    raise SystemExit(main())
