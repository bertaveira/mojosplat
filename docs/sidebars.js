// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: ['getting-started/quickstart', 'getting-started/mojo-library', 'getting-started/pytorch'],
    },
    'how-rendering-works',
    {
      type: 'category',
      label: 'Architecture',
      link: {type: 'doc', id: 'architecture/overview'},
      items: ['architecture/overview', 'architecture/adding-a-method', 'architecture/data-and-state'],
    },
  ],
};

export default sidebars;
