/**
 * PikMatrix Accessible Component System
 * Strict Anti-Slop: Zero Native <select>, Full 8-State Keyboard & ARIA Controls
 */

(function () {
  'use strict';

  function initCustomSelect(wrapper) {
    if (wrapper.dataset.initialized === 'true') return;
    wrapper.dataset.initialized = 'true';

    const trigger = wrapper.querySelector('.custom-select-trigger');
    const popover = wrapper.querySelector('.custom-select-popover');
    const hiddenInput = wrapper.querySelector('input[type="hidden"]');
    const labelSpan = trigger ? trigger.querySelector('.custom-select-label') : null;
    const options = popover ? Array.from(popover.querySelectorAll('.custom-select-option')) : [];

    if (!trigger || !popover || !hiddenInput) return;

    let isOpen = false;
    let focusedIndex = options.findIndex((opt) => opt.getAttribute('aria-selected') === 'true');
    if (focusedIndex === -1) focusedIndex = 0;

    function openDropdown() {
      isOpen = true;
      trigger.setAttribute('aria-expanded', 'true');
      popover.classList.add('open');
      if (options[focusedIndex]) {
        options[focusedIndex].focus();
      }
    }

    function closeDropdown(focusTrigger = true) {
      isOpen = false;
      trigger.setAttribute('aria-expanded', 'false');
      popover.classList.remove('open');
      if (focusTrigger) {
        trigger.focus();
      }
    }

    function selectOption(opt) {
      const val = opt.getAttribute('data-value') || opt.textContent.trim();
      const text = opt.getAttribute('data-label') || opt.textContent.trim();

      hiddenInput.value = val;
      if (labelSpan) labelSpan.textContent = text;

      options.forEach((o) => {
        o.setAttribute('aria-selected', o === opt ? 'true' : 'false');
        o.tabIndex = o === opt ? 0 : -1;
      });

      // Dispatch change event on the hidden input so onchange handlers fire
      const evt = new Event('change', { bubbles: true });
      hiddenInput.dispatchEvent(evt);

      closeDropdown();
    }

    trigger.addEventListener('click', function (e) {
      e.stopPropagation();
      if (isOpen) {
        closeDropdown();
      } else {
        openDropdown();
      }
    });

    trigger.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!isOpen) {
          openDropdown();
        }
      } else if (e.key === 'Escape' && isOpen) {
        e.preventDefault();
        closeDropdown();
      }
    });

    options.forEach((opt, idx) => {
      opt.addEventListener('click', function (e) {
        e.stopPropagation();
        focusedIndex = idx;
        selectOption(opt);
      });

      opt.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          focusedIndex = (focusedIndex + 1) % options.length;
          options[focusedIndex].focus();
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          focusedIndex = (focusedIndex - 1 + options.length) % options.length;
          options[focusedIndex].focus();
        } else if (e.key === 'Home') {
          e.preventDefault();
          focusedIndex = 0;
          options[focusedIndex].focus();
        } else if (e.key === 'End') {
          e.preventDefault();
          focusedIndex = options.length - 1;
          options[focusedIndex].focus();
        } else if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          selectOption(opt);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          closeDropdown();
        }
      });
    });

    document.addEventListener('click', function (e) {
      if (isOpen && !wrapper.contains(e.target)) {
        closeDropdown(false);
      }
    });
  }

  function initAllSelects() {
    document.querySelectorAll('.custom-select-wrapper').forEach(initCustomSelect);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllSelects);
  } else {
    initAllSelects();
  }

  // HTMX content swap support
  document.addEventListener('htmx:afterSwap', initAllSelects);

  window.PikMatrixComponents = {
    initCustomSelect,
    initAllSelects,
  };
})();
