import os

from .app import app


def _to_bool(value: str) -> bool:
	return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
	host = os.getenv("HOST", "0.0.0.0")
	port = int(os.getenv("PORT", "5000"))
	debug = _to_bool(os.getenv("DEBUG", "false"))
	app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
	main()
