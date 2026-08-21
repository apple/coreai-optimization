// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-Clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


// Runtime behaviour for the documentation version picker.
//
// Three jobs, all of which no-op on a doc set built without DOCS_VERSIONS (the
// _templates/components/nav-versions.html override renders nothing, so there is
// no picker to find):
//
// 1. Open and close the dropdown on click, rather than the theme's hover-only
//    reveal — so it works from the keyboard and on touch.
// 2. Refresh the version list from the site's versions.json. The list is rendered
//    into every page at build time, so an already-published doc set would
//    otherwise never learn about releases cut after it.
// 3. Warn when the reader is not on the newest version, with a link to it.
//
// Written in the same plain-ES5-with-const style as the sibling _static scripts.

// The site root is derived from this script's own URL, which is always
// `<site root>/<version>/_static/version-switcher.js` — so the root is three
// levels up, whatever the version is called and however deep the current page
// sits. Read at load time because document.currentScript is only set while the
// script is being evaluated, not later inside a callback.
//
// Deliberately not derived from DOCUMENTATION_OPTIONS.VERSION: that is Sphinx's
// `release` field, which exists to describe the project version. If anyone ever
// set it to a PEP 440 string (0.3.0) rather than the directory name (v0.3.0),
// matching against it would silently stop resolving.
const SITE_ROOT = document.currentScript
  ? new URL("../../", document.currentScript.src).href
  : null;

// Label suffix conf.py's version_label() appends to the development version.
// Stripped for the banner text, where "main (development)" reads awkwardly.
const DEV_SUFFIX = " (development)";

// Upgrade each menu entry from "that version's front page" to "this page in that
// version", where it exists. Runs once, on first open, so a reader who never
// touches the picker pays nothing and the cost does not grow with each release.
function resolveMenuLinks(picker) {
  const pending = picker.querySelectorAll("a[data-version-root]");
  pending.forEach(function (link) {
    const root = link.dataset.versionRoot;
    // Clear the marker first so a second open cannot re-probe the same entry.
    delete link.dataset.versionRoot;
    equivalentPageIn(root).then(function (href) {
      link.href = href;
    });
  });
}

function setupMenu(picker) {
  const button = picker.querySelector(".js-version-menu");
  const menu = picker.querySelector(".nav-versions-choices");
  if (!button || !menu) {
    return;
  }

  function close() {
    button.setAttribute("aria-expanded", "false");
  }

  button.addEventListener("click", function (event) {
    event.stopPropagation();
    const open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", open ? "false" : "true");
    if (!open) {
      resolveMenuLinks(picker);
    }
  });

  // Click anywhere else, or Escape, dismisses it — the behaviour a reader
  // expects from a menu and what the hover-only original could not offer.
  document.addEventListener("click", function (event) {
    if (!picker.contains(event.target)) {
      close();
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      close();
    }
  });
}

function renderVersions(picker, versions) {
  const current = picker.querySelector(".js-version-menu span");
  const currentLabel = current ? current.textContent : null;
  const list = document.createElement("ul");

  versions.forEach(function (entry) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.setAttribute("role", "menuitem");
    // Assign via properties, not innerHTML, so a label from versions.json is
    // inserted as text and cannot inject markup.
    link.textContent = entry[0];
    link.href = entry[1];
    if (entry[0] === currentLabel) {
      link.setAttribute("aria-current", "true");
    } else {
      // Deep-link resolution is deferred to the first time the menu is opened
      // (see resolveMenuLinks): probing every version here would cost one request
      // per release on every page view, which grows with each release for readers
      // who never touch the picker.
      link.dataset.versionRoot = entry[1];
    }
    item.appendChild(link);
    list.appendChild(item);
  });

  const menu = picker.querySelector(".nav-versions-choices");
  // The override template puts nothing else inside the menu, so replacing its
  // children swaps the list without disturbing anything.
  menu.replaceChildren(list);
}

// Map the current page onto another version, so a reader deep in the docs lands
// on the same topic rather than that version's front page.
//
// Falls back to the version root when the page does not exist there — pages get
// added, renamed and removed between releases, and a 404 is a worse outcome than
// the landing page. Verified with a HEAD request rather than assumed, since only
// the server knows what that version actually contains.
function equivalentPageIn(versionHref) {
  if (!SITE_ROOT) {
    return Promise.resolve(versionHref);
  }
  // The path of the current page relative to its own doc set, e.g.
  // "quantization/config.html" from ".../v0.2.0/quantization/config.html".
  const relative = window.location.href.slice(SITE_ROOT.length).split("/").slice(1).join("/");
  if (!relative) {
    return Promise.resolve(versionHref);
  }
  const target = versionHref + relative;
  return fetch(target, { method: "HEAD" })
    .then(function (response) {
      return response.ok ? target : versionHref;
    })
    .catch(function () {
      return versionHref;
    });
}

// Announce when the doc set being read is not the newest release, in one of two
// ways: `main` documents unreleased work, and an older release has been
// superseded. Both reuse the theme's own `.announcement` element, so they inherit
// its sticky positioning and the --sy-s-banner-height offset the header and
// sidebars already account for — worth far more than hand-rolling a bar and
// re-deriving those offsets.
function showVersionBanner(versions, currentLabel) {
  if (document.querySelector(".announcement")) {
    return; // a real announcement is configured; don't stack banners on it
  }

  // versions[0] is `main`; the newest actual release is the first entry that
  // isn't it. Ordering comes from assemble_versioned_site.sort_versions().
  const newest = versions.find(function (entry) {
    return entry[0].indexOf(DEV_SUFFIX) === -1;
  });
  // Nothing useful to point at until at least one release exists.
  if (!newest || newest[0] === currentLabel) {
    return;
  }

  const development = currentLabel.indexOf(DEV_SUFFIX) !== -1;
  const banner = document.createElement("div");
  // Development is a neutral heads-up; a superseded release is a caution. The
  // two get different palettes in custom.css.
  banner.className = development
    ? "announcement version-banner version-banner-dev"
    : "announcement version-banner version-banner-old";

  const inner = document.createElement("div");
  inner.className = "announcement-inner";

  const text = document.createElement("p");
  const emphasis = document.createElement("strong");
  if (development) {
    emphasis.textContent = "in-development docs";
    text.append("These are ", emphasis, " and may describe unreleased features. ");
  } else {
    emphasis.textContent = "version " + currentLabel;
    text.append("These docs are for ", emphasis, ", which is no longer the latest. ");
  }

  const link = document.createElement("a");
  link.className = "version-banner-action";
  // Point at the version root immediately so the link is never dead, then
  // upgrade it to this page's counterpart once the HEAD probe resolves.
  link.href = newest[1];
  link.textContent = "Go to " + newest[0];
  equivalentPageIn(newest[1]).then(function (href) {
    link.href = href;
  });
  text.append(link);

  const close = document.createElement("button");
  close.className = "announcement-close";
  close.setAttribute("aria-label", "Close notification");
  close.innerHTML = '<i class="i-lucide close"></i>';

  inner.append(text, close);
  banner.appendChild(inner);
  document.body.prepend(banner);

  // The theme's own script only wires the close button for a banner present at
  // load, so this one manages its own teardown and height variable.
  const style = document.createElement("style");
  const setHeight = function () {
    style.textContent = ":root{--sy-s-banner-height:" + banner.clientHeight + "px}";
  };
  document.head.appendChild(style);
  setHeight();
  window.addEventListener("resize", setHeight);
  close.addEventListener("click", function () {
    banner.remove();
    style.remove();
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const picker = document.querySelector("#version-picker");
  if (!picker) {
    return;
  }
  setupMenu(picker);

  if (!SITE_ROOT) {
    return;
  }
  fetch(SITE_ROOT + "versions.json", { cache: "no-cache" })
    .then(function (response) {
      return response.ok ? response.json() : Promise.reject(response.status);
    })
    .then(function (versions) {
      if (!Array.isArray(versions) || versions.length === 0) {
        return;
      }
      renderVersions(picker, versions);
      const current = picker.querySelector(".js-version-menu span");
      if (current) {
        showVersionBanner(versions, current.textContent);
      }
    })
    // Leave the build-time list in place when versions.json is missing or
    // unreachable — a stale picker beats an empty one. The banner is skipped
    // too, since without the manifest there is no way to know what is newest.
    .catch(function () {});
});
