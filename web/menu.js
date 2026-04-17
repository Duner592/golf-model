(() => {
    const focusableSelectors = [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])'
    ].join(', ');

    const quickLinks = [
        { label: 'Home', href: 'index.html' },
        {
            label: 'Tournament Predictions',
            href: 'pga.html',
            children: [
                { label: 'PGA Tour', href: 'pga.html' },
                { label: 'DP World Tour', href: 'euro.html' },
                { label: 'Prediction Archives', href: 'archive.html' },
                { label: 'Accuracy Dashboards', href: 'archive_accuracy.html' }
            ]
        },
        {
            label: 'Schedules',
            href: 'schedule.html',
            children: [
                { label: 'Tour Schedule', href: 'schedule.html' },
                { label: 'PGA Schedule', href: 'pga_schedule.html' },
                { label: 'DP World Schedule', href: 'euro_schedule.html' }
            ]
        },
        {
            label: 'Betting History',
            href: 'betting_analytics.html',
            children: [
                { label: 'Analytics', href: 'betting_analytics.html' },
                { label: 'Overall Data', href: 'spreadsheet.html' },
                { label: 'ROI% Data', href: 'roi.html' },
                { label: 'Yearly 2026', href: 'spreadsheet.html?year=2026' },
                { label: 'Yearly 2025', href: 'spreadsheet.html?year=2025' },
                { label: 'Yearly 2024', href: 'spreadsheet.html?year=2024' },
                { label: 'Yearly 2023', href: 'spreadsheet.html?year=2023' }
            ]
        },
        {
            label: 'Additional Data',
            href: 'course_details.html',
            children: [
                { label: 'Course Details', href: 'course_details.html' },
                { label: 'Notes', href: 'notes.html' },
                { label: 'Links & Tools', href: 'links.html' }
            ]
        }
    ];

    let lastFocusedElement = null;
    let menuObserverStarted = false;
    let quickNavElement = null;
    let openDropdown = null;
    let dropdownIdCounter = 0;

    function normalizeRoute(route) {
        if (!route) {
            return 'index.html';
        }
        const withoutHash = route.split('#')[0];
        const [path] = withoutHash.split('?');
        const segments = path.split('/').filter(Boolean);
        if (!segments.length) {
            return 'index.html';
        }
        const last = segments[segments.length - 1].toLowerCase();
        return last || 'index.html';
    }

    function isCurrentRoute(href, matches = []) {
        const candidates = new Set(matches.map(normalizeRoute));
        candidates.add(normalizeRoute(href));
        const current = normalizeRoute(window.location.pathname);
        if (window.location.pathname === '/' && candidates.has('index.html')) {
            return true;
        }
        return candidates.has(current);
    }

    function isMenuLinkActive(href) {
        if (!href) {
            return false;
        }
        const [pathPart, queryPart = ''] = href.split('?');
        const normalizedHref = normalizeRoute(pathPart);
        const currentPath = normalizeRoute(window.location.pathname);
        if (normalizedHref !== currentPath) {
            return false;
        }
        if (!queryPart) {
            return true;
        }
        const targetParams = new URLSearchParams(queryPart);
        const currentParams = new URLSearchParams(window.location.search || '');
        for (const [key, value] of targetParams.entries()) {
            if (currentParams.get(key) !== value) {
                return false;
            }
        }
        return true;
    }

    function getMenuElements() {
        return {
            menu: document.getElementById('menu'),
            overlay: document.getElementById('overlay'),
            body: document.body,
            hamburger: document.querySelector('.hamburger')
        };
    }

    function ensureHamburgerAttributes() {
        const { hamburger } = getMenuElements();
        if (!hamburger || hamburger.tagName !== 'BUTTON') {
            return;
        }
        hamburger.type = 'button';
        hamburger.setAttribute('aria-haspopup', 'true');
        if (!hamburger.hasAttribute('aria-controls')) {
            hamburger.setAttribute('aria-controls', 'menu');
        }
        if (!hamburger.hasAttribute('aria-expanded')) {
            hamburger.setAttribute('aria-expanded', 'false');
        }
    }

    function isMenuOpen(menu) {
        return menu?.classList.contains('show');
    }

    function focusFirstElement(menu) {
        if (!menu) {
            return;
        }
        const focusable = menu.querySelector(focusableSelectors);
        if (focusable instanceof HTMLElement) {
            focusable.focus();
            return;
        }
        if (!menu.hasAttribute('tabindex')) {
            menu.setAttribute('tabindex', '-1');
        }
        menu.focus();
    }

    function toggleDisclosure(button, force) {
        if (!(button instanceof HTMLElement)) {
            return;
        }
        const targetId = button.getAttribute('data-menu-toggle');
        if (!targetId) {
            return;
        }
        const content = document.getElementById(targetId);
        if (!content) {
            return;
        }
        const shouldExpand = force !== undefined ? force : button.getAttribute('aria-expanded') !== 'true';
        button.setAttribute('aria-expanded', String(shouldExpand));
        const arrow = button.querySelector('.arrow');
        if (arrow) {
            arrow.textContent = shouldExpand ? '▼' : '▶';
        }
        if (shouldExpand) {
            content.removeAttribute('hidden');
        } else {
            content.setAttribute('hidden', '');
        }
    }

    function trapFocus(event, menu) {
        if (!menu) {
            return;
        }
        const focusableElements = Array.from(menu.querySelectorAll(focusableSelectors))
            .filter((el) => el instanceof HTMLElement && !el.hasAttribute('disabled') && el.getAttribute('aria-hidden') !== 'true');

        if (!focusableElements.length) {
            return;
        }

        const first = focusableElements[0];
        const last = focusableElements[focusableElements.length - 1];

        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }

    function closeCurrentDropdown(options = {}) {
        if (!openDropdown) {
            return;
        }
        const { toggle, dropdown } = openDropdown;
        toggle.classList.remove('site-header__quick-toggle--open');
        toggle.setAttribute('aria-expanded', 'false');
        dropdown.setAttribute('hidden', '');
        if (options.focus && toggle instanceof HTMLElement) {
            toggle.focus({ preventScroll: true });
        }
        openDropdown = null;
    }

    function openQuickDropdown(toggle, dropdown) {
        if (!(toggle instanceof HTMLElement) || !(dropdown instanceof HTMLElement)) {
            return;
        }
        if (openDropdown && openDropdown.toggle === toggle) {
            closeCurrentDropdown();
            return;
        }
        closeCurrentDropdown();
        toggle.classList.add('site-header__quick-toggle--open');
        toggle.setAttribute('aria-expanded', 'true');
        dropdown.removeAttribute('hidden');
        openDropdown = { toggle, dropdown };
    }

    function toggleMenu(force) {
        const { menu, overlay, body, hamburger } = getMenuElements();
        if (!menu || !overlay || !body || !hamburger) {
            return;
        }
        const shouldOpen = force === undefined ? !isMenuOpen(menu) : force;
        if (shouldOpen) {
            lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
            closeCurrentDropdown();
            menu.classList.add('show');
            overlay.classList.add('show');
            overlay.setAttribute('aria-hidden', 'false');
            body.classList.add('menu-open');
            hamburger.setAttribute('aria-expanded', 'true');
            requestAnimationFrame(() => focusFirstElement(menu));
        } else {
            menu.classList.remove('show');
            overlay.classList.remove('show');
            overlay.setAttribute('aria-hidden', 'true');
            body.classList.remove('menu-open');
            hamburger.setAttribute('aria-expanded', 'false');
            if (lastFocusedElement instanceof HTMLElement) {
                lastFocusedElement.focus({ preventScroll: true });
                lastFocusedElement = null;
            }
        }
    }

    function buildBrand() {
        const brand = document.createElement('div');
        brand.className = 'site-header__brand';
        const link = document.createElement('a');
        link.className = 'site-header__brand-link';
        link.href = 'index.html';
        link.setAttribute('aria-label', 'Go to homepage');
        const logo = document.querySelector('.logo');
        if (logo instanceof HTMLImageElement) {
            const clone = logo.cloneNode(true);
            clone.classList.remove('logo');
            clone.classList.add('site-header__brand-image');
            link.appendChild(clone);
            const wrapper = logo.closest('.logo-wrapper');
            if (wrapper) {
                wrapper.remove();
            } else {
                logo.remove();
            }
        } else {
            link.textContent = 'Eds Golf Model';
        }
        brand.appendChild(link);
        return brand;
    }

    function createDropdown(toggle, children, parentMatches = []) {
        const dropdown = document.createElement('div');
        dropdown.className = 'site-header__dropdown';
        dropdown.setAttribute('role', 'menu');
        dropdown.setAttribute('hidden', '');
        children.forEach((child) => {
            const anchor = document.createElement('a');
            anchor.className = 'site-header__dropdown-link';
            anchor.href = child.href;
            anchor.textContent = child.label;
            anchor.setAttribute('role', 'menuitem');
            anchor.dataset.quickChild = 'true';
            if (Array.isArray(child.matches)) {
                parentMatches.push(...child.matches);
            }
            anchor.addEventListener('click', () => {
                closeCurrentDropdown();
            });
            dropdown.appendChild(anchor);
        });
        return dropdown;
    }

    function buildQuickNav() {
        const nav = document.createElement('div');
        nav.className = 'site-header__quick-nav';
        nav.setAttribute('role', 'navigation');
        nav.setAttribute('aria-label', 'Primary navigation');
        quickLinks.forEach((item) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'site-header__quick-item';
            wrapper.dataset.quickItem = 'true';
            if (item.href) {
                wrapper.dataset.primaryHref = item.href;
            }
            const hasChildren = Array.isArray(item.children) && item.children.length > 0;
            if (hasChildren) {
                dropdownIdCounter += 1;
                const dropdownId = `site-header-dropdown-${dropdownIdCounter}`;
                const toggle = document.createElement('button');
                toggle.className = 'site-header__quick-toggle';
                toggle.type = 'button';
                toggle.textContent = item.label;
                toggle.setAttribute('aria-haspopup', 'true');
                toggle.setAttribute('aria-expanded', 'false');
                toggle.setAttribute('aria-controls', dropdownId);
                if (item.href) {
                    toggle.dataset.primaryHref = item.href;
                }
                toggle.addEventListener('click', () => {
                    const dropdown = document.getElementById(dropdownId);
                    openQuickDropdown(toggle, dropdown);
                });
                wrapper.appendChild(toggle);
                const dropdown = createDropdown(toggle, item.children, item.matches);
                dropdown.id = dropdownId;
                wrapper.appendChild(dropdown);
            } else {
                const anchor = document.createElement('a');
                anchor.className = 'site-header__quick-link';
                anchor.href = item.href;
                anchor.textContent = item.label;
                anchor.addEventListener('click', () => {
                    closeCurrentDropdown();
                });
                wrapper.appendChild(anchor);
            }
            nav.appendChild(wrapper);
        });
        quickNavElement = nav;
        return nav;
    }

    function injectSiteHeader() {
        if (document.querySelector('.site-header')) {
            return;
        }
        const menuContainer = document.querySelector('.menu-container');
        if (!menuContainer) {
            return;
        }
        const header = document.createElement('header');
        header.className = 'site-header';
        const inner = document.createElement('div');
        inner.className = 'site-header__inner';

        const actions = document.createElement('div');
        actions.className = 'site-header__actions';
        actions.appendChild(menuContainer);

        inner.appendChild(actions);
        inner.appendChild(buildQuickNav());
        inner.appendChild(buildBrand());

        header.appendChild(inner);

        const overlay = document.getElementById('overlay');
        if (overlay) {
            overlay.insertAdjacentElement('afterend', header);
        } else if (document.body.firstChild) {
            document.body.insertBefore(header, document.body.firstChild);
        } else {
            document.body.appendChild(header);
        }
    }

    function applyMenuActiveState(menu) {
        if (!menu) {
            return;
        }
        menu.querySelectorAll('.menu-link').forEach((link) => {
            if (!(link instanceof HTMLAnchorElement)) {
                return;
            }
            const href = link.getAttribute('href') || '';
            const active = isMenuLinkActive(href);
            link.classList.toggle('menu-link--active', active);
            if (active) {
                link.setAttribute('aria-current', 'page');
            } else {
                link.removeAttribute('aria-current');
            }
        });
    }

    function applyQuickNavActiveState() {
        if (!quickNavElement) {
            return;
        }
        quickNavElement.querySelectorAll('.site-header__quick-item').forEach((item) => {
            const toggle = item.querySelector('.site-header__quick-toggle');
            const link = item.querySelector('.site-header__quick-link');
            const dropdownLinks = item.querySelectorAll('.site-header__dropdown-link');
            let active = false;

            dropdownLinks.forEach((anchor) => {
                if (!(anchor instanceof HTMLAnchorElement)) {
                    return;
                }
                const href = anchor.getAttribute('href') || '';
                const isActive = isMenuLinkActive(href);
                anchor.classList.toggle('is-active', isActive);
                if (isActive) {
                    anchor.setAttribute('aria-current', 'page');
                } else {
                    anchor.removeAttribute('aria-current');
                }
                active = active || isActive;
            });

            if (toggle) {
                const primaryHref = toggle.dataset.primaryHref || item.dataset.primaryHref || '';
                if (primaryHref && isMenuLinkActive(primaryHref)) {
                    active = true;
                }
                toggle.classList.toggle('site-header__quick-toggle--active', active);
                if (active) {
                    toggle.setAttribute('aria-current', 'page');
                } else {
                    toggle.removeAttribute('aria-current');
                }
            }

            if (link instanceof HTMLAnchorElement) {
                const href = link.getAttribute('href') || '';
                const linkActive = isMenuLinkActive(href);
                link.classList.toggle('site-header__quick-link--active', linkActive);
                if (linkActive) {
                    link.setAttribute('aria-current', 'page');
                } else {
                    link.removeAttribute('aria-current');
                }
            }
        });
    }

    function hydrateMenuContent(menu) {
        if (!menu) {
            return;
        }
        menu.querySelectorAll('[data-menu-toggle]').forEach((button) => {
            if (!(button instanceof HTMLElement)) {
                return;
            }
            const isExpanded = button.getAttribute('aria-expanded') === 'true';
            toggleDisclosure(button, isExpanded);
        });
        applyMenuActiveState(menu);
        applyQuickNavActiveState();
    }

    function initializeMenuObserver() {
        if (menuObserverStarted) {
            return;
        }
        const { menu } = getMenuElements();
        if (!menu) {
            return;
        }
        const observer = new MutationObserver((mutations, obs) => {
            if (menu.children.length) {
                hydrateMenuContent(menu);
                obs.disconnect();
            }
        });
        observer.observe(menu, { childList: true, subtree: true });
        menuObserverStarted = true;
    }

    window.toggleMenu = toggleMenu;

    document.addEventListener('click', (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }

        if (openDropdown) {
            const { toggle, dropdown } = openDropdown;
            if (!toggle.contains(target) && !dropdown.contains(target)) {
                closeCurrentDropdown();
            }
        }

        const toggleButton = target.closest('[data-menu-toggle]');
        if (toggleButton instanceof HTMLElement) {
            event.preventDefault();
            toggleDisclosure(toggleButton);
            return;
        }

        const link = target.closest('.menu-link');
        if (link) {
            toggleMenu(false);
        }
    });

    document.addEventListener('keydown', (event) => {
        const { menu } = getMenuElements();

        if (event.key === 'Escape') {
            if (openDropdown) {
                event.preventDefault();
                const toggle = openDropdown.toggle;
                closeCurrentDropdown({ focus: true });
                if (toggle instanceof HTMLElement) {
                    toggle.focus({ preventScroll: true });
                }
                return;
            }
        }

        if (!menu || !isMenuOpen(menu)) {
            return;
        }

        if (event.key === 'Escape') {
            event.preventDefault();
            toggleMenu(false);
            return;
        }

        if (event.key === 'Tab') {
            trapFocus(event, menu);
        }
    });

    document.addEventListener('DOMContentLoaded', () => {
        const { overlay, body, menu } = getMenuElements();
        if (overlay && !overlay.hasAttribute('aria-hidden')) {
            overlay.setAttribute('aria-hidden', 'true');
        }
        ensureHamburgerAttributes();
        injectSiteHeader();
        initializeMenuObserver();
        if (menu && menu.children.length) {
            hydrateMenuContent(menu);
        }
        applyQuickNavActiveState();
        if (body && !body.classList.contains('has-fixed-menu')) {
            body.classList.add('has-fixed-menu');
        }
    });

    window.addEventListener('resize', () => {
        if (!openDropdown) {
            return;
        }
        const { dropdown } = openDropdown;
        if (dropdown instanceof HTMLElement) {
            dropdown.setAttribute('hidden', '');
            requestAnimationFrame(() => {
                dropdown.removeAttribute('hidden');
            });
        }
    });
})();
