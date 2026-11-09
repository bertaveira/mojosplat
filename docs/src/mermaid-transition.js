// Mermaid imports only `select` from d3, so bundlers drop d3-transition, the
// module that adds `.transition()` to selections. Node tooltips need it.
import 'd3-transition';
