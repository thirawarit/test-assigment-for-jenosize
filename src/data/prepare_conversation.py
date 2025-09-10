import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

class LevelFormatter(logging.Formatter):
    """Custom formatter: format depends on log level."""
    FORMATS = {
        logging.ERROR:  logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                                          datefmt="%m/%d/%Y %H:%M:%S"),
        'DEFAULT': logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                     datefmt="%m/%d/%Y %H:%M:%S",)
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return formatter.format(record)
    
# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(LevelFormatter())
logging.basicConfig(
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[handler]
)
logger = logging.getLogger(__name__)

def generate_conversation(dataset: List[Dict[str, Any]], save_path: Union[str | Path]):

    system_prompt = (
        "You are a helpful assistant."
    )

    instruction_prompt = (
        "* Analyse the given article. \n"
        "* Create articles about trends and future ideas for businesses as a markdown format \n"
        "* No reasoning, no explanation. \n"
        "assistant: \n"
        "# <TOPIC> \n"
        "    <DESCRIPTION> \n"
        "## <SUB TOPIC> \n"
        "    <DESCRIPTION> \n\n"
        "<|RAW_TEXT|> \n"
        "topic category: {topic_category} \n"
        "industry: {industry} \n"
        "target audience: {target_audience} \n"
        "source of website data or document: {website} \n"
        "SEO keywords: {seo_keywords} \n"
        "<|/RAW_TEXT|> "
    )

    if isinstance(save_path, str):
        save_path = Path(save_path)
    if not save_path.parent.exists():
        logging.info("Creating save directory at: %s", save_path.parent)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = save_path.open('w')

    list_conv = [] # A list of conversations
    for d in dataset:

        # For JSONL format.
        if isinstance(d, str):
            d = json.loads(d)

        conversations = []

        system = {'from': 'system', 'value': system_prompt}
        conversations.append(system)
        
        human = {'from': 'human', 'value': instruction_prompt.format(**d)}
        conversations.append(human)

        gpt = {'from': 'gpt', 'value': d['content']}
        conversations.append(gpt)
        list_conv.append({'conversations': conversations})

        if save_path.suffix == '.jsonl':
            writer.write(json.dumps({'conversations': conversations}) + '\n')

    if save_path.suffix == '.json':
        json.dump(list_conv, writer)

    writer.close()


def main():
    json_file_path = Path(args.input_path)
    if not json_file_path.exists():
        raise FileExistsError(f"`--input-path` doesn't exist, please check the path again, ({json_file_path.resolve()})")

    if json_file_path.suffix == '.json':
        dataset = json.load(json_file_path.open('r'))
    elif json_file_path.suffix == '.jsonl':
        dataset = json_file_path.open('r').readlines()
    else:
        raise NameError(f'Support both extensions ".json" or ".jsonl", but --input-path {json_file_path.resolve()}')
    

    save_path = Path(args.save_path)
    if not save_path.suffix in ('.json', '.jsonl'):
        raise NameError(f'Support both extensions ".json" or ".jsonl", but --save-path {save_path.resolve()}')
    
    generate_conversation(dataset=dataset, save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='test', type=str, nargs=1, help='description (default: test)')
    parser.add_argument('--input-path', required=True, type=str, help='a file name in the path that load data.')
    parser.add_argument('--save-path', required=True, type=str, help='a file name in the path that you want to save at.')
    args = parser.parse_args()
    main()