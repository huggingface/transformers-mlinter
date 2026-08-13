// Client-side filter for the rule index table.
//
// The index is one long table by design — 57 rules with no categories to browse by — so the fast path
// to a rule is typing part of its code or description. This runs only on the rule index; every other
// page exits at the first guard.
//
// The site's search box (top left) covers full-text search across all pages. This is the narrower
// tool: it filters the table in place so you keep the Default and Scope columns while narrowing.
(function () {
  "use strict";

  var input = document.getElementById("rule-filter");
  var table = document.getElementById("rule-table");
  var counter = document.getElementById("rule-filter-count");
  if (!input || !table) {
    return;
  }

  var rows = Array.prototype.slice.call(table.tBodies[0].rows);

  // Cache each row's searchable text once. Reading textContent per row per keystroke is what makes a
  // naive version of this feel laggy on a long table.
  var haystacks = rows.map(function (row) {
    return row.textContent.toLowerCase();
  });

  // Mark the disabled-by-default rules so the exception is visible without the generator having to
  // emit styling into the Markdown table.
  rows.forEach(function (row) {
    var cell = row.cells[2];
    if (cell && cell.textContent.trim() === "Disabled") {
      cell.classList.add("rule-default-disabled");
    }
  });

  function report(visible) {
    if (!counter) {
      return;
    }
    counter.textContent =
      visible === rows.length
        ? rows.length + " rules"
        : visible + " of " + rows.length + " rules";
  }

  function apply(query) {
    // Every whitespace-separated term must match, so "trf0 config" narrows rather than widens.
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    var visible = 0;

    rows.forEach(function (row, index) {
      var matches = terms.every(function (term) {
        return haystacks[index].indexOf(term) !== -1;
      });
      row.hidden = !matches;
      if (matches) {
        visible += 1;
      }
    });

    report(visible);
  }

  input.addEventListener("input", function () {
    apply(input.value);
  });

  // Escape clears, which is the reflex for a filter box.
  input.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && input.value !== "") {
      event.preventDefault();
      input.value = "";
      apply("");
    }
  });

  // ?q=… makes a filtered view linkable — handy for pointing someone at "every rule mentioning
  // hidden_act" without them having to retype it.
  var initial = new URLSearchParams(window.location.search).get("q");
  if (initial) {
    input.value = initial;
  }
  apply(input.value);
})();
