// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'MojoSplat',
  tagline: '3D Gaussian Splatting in pure Mojo',
  url: 'https://bertaveira.github.io',
  baseUrl: '/mojosplat/',
  organizationName: 'bertaveira',
  projectName: 'mojosplat',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    format: 'detect', // .md is plain Markdown: no MDX parsing of < and { in prose
  },
  themes: ['@docusaurus/theme-mermaid'],
  clientModules: ['./src/mermaid-transition.js'],
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/bertaveira/mojosplat/edit/main/docs/',
        },
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'MojoSplat',
        items: [
          {type: 'doc', docId: 'getting-started/quickstart', position: 'left', label: 'Getting started'},
          {type: 'doc', docId: 'how-rendering-works', position: 'left', label: 'How rendering works'},
          {type: 'doc', docId: 'architecture/overview', position: 'left', label: 'Architecture'},
          {type: 'doc', docId: 'results', position: 'left', label: 'Methods and results'},
          {href: 'https://github.com/bertaveira/mojosplat', label: 'GitHub', position: 'right'},
        ],
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'python'],
      },
    }),
};

export default config;
