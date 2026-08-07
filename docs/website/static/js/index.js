window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Copied!';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            button.classList.add('copied');
            copyText.textContent = 'Copied!';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Data sample explorer: one panel at a time, with the videos of hidden panels paused.
// Without JavaScript, every panel stays visible, so the samples remain readable.
(function () {
    const section = document.getElementById('samples');
    if (!section) return;

    const chips = Array.prototype.slice.call(section.querySelectorAll('.sample-chip'));
    const panels = Array.prototype.slice.call(section.querySelectorAll('.sample-panel'));
    if (!chips.length || !panels.length) return;

    // Only now that the script runs do we collapse the list to a single visible panel.
    section.classList.add('is-interactive');

    function select(sampleId) {
        chips.forEach(function (chip) {
            const active = chip.dataset.sample === sampleId;
            chip.classList.toggle('is-active', active);
            chip.setAttribute('aria-selected', active ? 'true' : 'false');
        });

        panels.forEach(function (panel) {
            const active = panel.dataset.sample === sampleId;
            panel.classList.toggle('is-active', active);

            const video = panel.querySelector('video');
            if (!video) return;
            if (active) {
                const play = video.play();
                if (play && typeof play.catch === 'function') {
                    play.catch(function () { /* autoplay blocked; controls remain */ });
                }
            } else {
                video.pause();
            }
        });
    }

    chips.forEach(function (chip) {
        chip.addEventListener('click', function () {
            select(chip.dataset.sample);
        });
    });

    // Arrow keys move between samples, as expected for a tab list.
    section.querySelector('.sample-chips').addEventListener('keydown', function (event) {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        const current = chips.findIndex(function (chip) { return chip.classList.contains('is-active'); });
        if (current === -1) return;
        const step = event.key === 'ArrowRight' ? 1 : -1;
        const next = chips[(current + step + chips.length) % chips.length];
        select(next.dataset.sample);
        next.focus();
        event.preventDefault();
    });
})();
