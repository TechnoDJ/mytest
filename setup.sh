mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

apt-get update && apt-get install -y python3-distutils
