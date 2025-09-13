input_prompt_template = (
    "* Analyse the given article. \n"
    "* Create articles about trends and future ideas for businesses as a markdown format \n"
    # "* No reasoning, no explanation. \n"
    "assistant: \n"
    "# <TOPIC> \n"
    "    <YOUR DESCRIPTION> \n"
    "## <SUB TOPIC> \n"
    "    <YOUR DESCRIPTION> \n"
    "\n\n"
    "<|RAW_TEXT|> \n"
    "topic category: {topic_category} \n"
    "industry: {industry} \n"
    "target audience: {target_audience} \n"
    "source of website data or document: {website} \n"
    "SEO keywords: {seo_keywords} \n"
    "<|/RAW_TEXT|> "
)

def build_input_prompt(items):
    return input_prompt_template.format(**items)