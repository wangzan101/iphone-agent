import sys

from iphone_agent.cli.commands import Session, dispatch


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        try:
            from iphone_agent.cli.repl import repl
        except ImportError:
            print("REPL 需要安装 repl 依赖：pip install -e '.[repl]'", file=sys.stderr)
            return 2
        return repl(Session())
    return dispatch(Session(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
