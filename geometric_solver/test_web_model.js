"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(
  path.join(__dirname, "wind_turbine_vector_ui.html"),
  "utf8",
);
const scriptStart = html.indexOf("<script>");
const scriptEnd = html.indexOf("</script>", scriptStart);
assert.ok(scriptStart >= 0 && scriptEnd > scriptStart, "script block is missing");

const script = html.slice(scriptStart + "<script>".length, scriptEnd);
const pureStart = script.indexOf("const rad =");
const pureEnd = script.indexOf("function selectedBladeIds");
assert.ok(pureStart >= 0 && pureEnd > pureStart, "pure model block is missing");

const context = {BETAS_DEG: {1: 0, 2: 120, 3: 240}};
vm.createContext(context);
vm.runInContext(
  `${script.slice(pureStart, pureEnd)}\n` +
  "globalThis.model = {orientationClass, projectDirections};",
  context,
);

assert.equal(context.model.orientationClass(0), "back");
assert.equal(context.model.orientationClass(90), "right");
assert.equal(context.model.orientationClass(135), "front_right");
assert.equal(context.model.orientationClass(225), "front_left");

const expected = [
  [-0.13865561209227384, 0.99034065918517],
  [-0.5867071878203843, -0.809799157668058],
  [0.9926065604281926, -0.12137634116627874],
];
const actual = context.model.projectDirections(20, 135, 15, -8);
actual.forEach((vector, index) => {
  assert.ok(vector.unit, `blade ${index + 1} has a degenerate projection`);
  assert.ok(Math.abs(vector.unit[0] - expected[index][0]) < 1e-12);
  assert.ok(Math.abs(vector.unit[1] - expected[index][1]) < 1e-12);
});

function element(value = "") {
  return {
    value,
    checked: true,
    dataset: {},
    style: {},
    children: [],
    textContent: "",
    classList: {toggle() {}},
    addEventListener() {},
    setAttribute() {},
    appendChild(child) { this.children.push(child); },
    replaceChildren(...children) { this.children = children; },
  };
}

const elements = {
  yaw: element("135"),
  yawNumber: element("135"),
  alpha: element("20"),
  alphaNumber: element("20"),
  pitch: element("15"),
  pitchNumber: element("15"),
  roll: element("-8"),
  rollNumber: element("-8"),
};
[
  "blades", "orientationMetric", "sideDistanceMetric", "visibleMetric",
  "sideWarning", "orientationBadge", "vectorOutput", "pythonOutput",
  "jsonOutput",
].forEach(id => { elements[id] = element(); });

const checkboxes = [1, 2, 3].map(id => {
  const checkbox = element();
  checkbox.dataset.blade = String(id);
  return checkbox;
});
const domContext = {
  console,
  document: {
    getElementById(id) { return elements[id] || element(); },
    querySelectorAll(selector) { return selector === "[data-blade]" ? checkboxes : []; },
    createElementNS() { return element(); },
  },
};
vm.createContext(domContext);
vm.runInContext(script, domContext);

assert.match(elements.orientationMetric.textContent, /front_right/);
assert.match(elements.pythonOutput.textContent, /solve_pose_from_unit_centerlines/);
assert.match(elements.pythonOutput.textContent, /camera_roll_deg=-8\.0/);
assert.ok(elements.blades.children.length >= 9, "blade drawing was not rendered");

console.log("Web UI model and render flow match the Python reference sample.");

